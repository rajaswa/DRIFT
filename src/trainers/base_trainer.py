import math
import os
import torch
from src.modules.optimizers import *
from src.modules.embeddings import *
from src.modules.schedulers import *
from src.modules.tokenizers import *
from src.modules.metrics import *
from src.modules.losses import *
from src.utils.misc import *
from src.utils.logger import Logger
from src.utils.mapper import configmapper
from src.utils.configuration import Config

from torch.utils.data import DataLoader
from tqdm import tqdm


@configmapper.map("trainers", "base")
class BaseTrainer:
    def __init__(self, config):
        self._config = config
        self.metrics = {
            configmapper.get_object("metrics", metric["type"]): metric["params"]
            for metric in self._config.main_config.metrics
        }
        self.train_config = self._config.train
        self.val_config = self._config.val
        self.log_label = self.train_config.log.log_label
        if self.train_config.log_and_val_interval is not None:
            self.val_log_together = True
        print("Logging with label: ", self.log_label)

    def train(self, model, train_dataset, val_dataset=None, logger=None):
        device = torch.device(self._config.main_config.device.name)
        model.to(device)
        optim_params = self.train_config.optimizer.params
        if optim_params:
            optimizer = configmapper.get_object(
                "optimizers", self.train_config.optimizer.type
            )(model.parameters(), **optim_params.as_dict())
        else:
            optimizer = configmapper.get_object(
                "optimizers", self.train_config.optimizer.type
            )(model.parameters())

        if self.train_config.scheduler is not None:
            scheduler_params = self.train_config.scheduler.params
            if scheduler_params:
                scheduler = configmapper.get_object(
                    "schedulers", self.train_config.scheduler.type
                )(optimizer, **scheduler_params.as_dict())
            else:
                scheduler = configmapper.get_object(
                    "schedulers", self.train_config.scheduler.type
                )(optimizer)

        criterion_params = self.train_config.criterion.params
        if criterion_params:
            criterion = configmapper.get_object(
                "losses", self.train_config.criterion.type
            )(**criterion_params.as_dict())
        else:
            criterion = configmapper.get_object(
                "losses", self.train_config.criterion.type
            )()
        if "custom_collate_fn" in dir(train_dataset):
            train_loader = DataLoader(
                dataset=train_dataset,
                collate_fn=train_dataset.custom_collate_fn,
                **self.train_config.loader_params.as_dict(),
            )
        else:
            train_loader = DataLoader(
                dataset=train_dataset, **self.train_config.loader_params.as_dict()
            )
        # train_logger = Logger(**self.train_config.log.logger_params.as_dict())

        max_epochs = self.train_config.max_epochs
        batch_size = self.train_config.loader_params.batch_size

        if self.val_log_together:
            val_interval = self.train_config.log_and_val_interval
            log_interval = val_interval
        else:
            val_interval = self.train_config.val_interval
            log_interval = self.train_config.log.log_interval

        if logger is None:
            train_logger = Logger(**self.train_config.log.logger_params.as_dict())
        else:
            train_logger = logger

        train_log_values = self.train_config.log.values.as_dict()

        best_score = (
            -math.inf if self.train_config.save_on.desired == "max" else math.inf
        )
        save_on_score = self.train_config.save_on.score
        best_step = -1
        best_model = None

        best_hparam_list = None
        best_hparam_name_list = None
        best_metrics_list = None
        best_metrics_name_list = None

        # print("\nTraining\n")
        # print(max_steps)

        global_step = 0
        for epoch in range(1, max_epochs + 1):
            print(
                "Epoch: {}/{}, Global Step: {}".format(epoch, max_epochs, global_step)
            )
            train_loss = 0
            val_loss = 0

            if self.train_config.label_type == "float":
                all_labels = torch.FloatTensor().to(device)
            else:
                all_labels = torch.LongTensor().to(device)

            all_outputs = torch.Tensor().to(device)

            train_scores = None
            val_scores = None

            pbar = tqdm(total=math.ceil(len(train_dataset) / batch_size))
            pbar.set_description("Epoch " + str(epoch))

            val_counter = 0

            for step, batch in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                inputs, labels = batch

                if self.train_config.label_type == "float":  ##Specific to Float Type
                    labels = labels.float()

                for key in inputs:
                    inputs[key] = inputs[key].to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(torch.squeeze(outputs, dim=1), labels)
                loss.backward()

                all_labels = torch.cat((all_labels, labels), 0)

                if self.train_config.label_type == "float":
                    all_outputs = torch.cat((all_outputs, outputs), 0)
                else:
                    all_outputs = torch.cat(
                        (all_outputs, torch.argmax(outputs, axis=1)), 0
                    )

                train_loss += loss.item()
                optimizer.step()

                if self.train_config.scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(train_loss / (step + 1))
                    else:
                        scheduler.step()

                # print(train_loss)
                # print(step+1)

                pbar.set_postfix_str(f"Train Loss: {train_loss /(step+1)}")
                pbar.update(1)

                global_step += 1

                # Need to check if we want global_step or local_step

                if val_dataset is not None and (global_step - 1) % val_interval == 0:
                    # print("\nEvaluating\n")
                    val_scores = self.val(
                        model,
                        val_dataset,
                        criterion,
                        device,
                        global_step,
                        train_logger,
                        train_log_values,
                    )

                    # save_flag = 0
                    if self.train_config.save_on is not None:

                        ## BEST SCORES UPDATING

                        train_scores = self.get_scores(
                            train_loss,
                            global_step,
                            self.train_config.criterion.type,
                            all_outputs,
                            all_labels,
                        )

                        best_score, best_step, save_flag = self.check_best(
                            val_scores, save_on_score, best_score, global_step
                        )

                        store_dict = {
                            "model_state_dict": model.state_dict(),
                            "best_step": best_step,
                            "best_score": best_score,
                            "save_on_score": save_on_score,
                        }

                        path = self.train_config.save_on.best_path.format(
                            self.log_label
                        )

                        self.save(store_dict, path, save_flag)

                        if save_flag and train_log_values["hparams"] is not None:
                            (
                                best_hparam_list,
                                best_hparam_name_list,
                                best_metrics_list,
                                best_metrics_name_list,
                            ) = self.update_hparams(
                                train_scores, val_scores, desc="best_val"
                            )
                # pbar.close()
                if (global_step - 1) % log_interval == 0:
                    # print("\nLogging\n")
                    train_loss_name = self.train_config.criterion.type
                    metric_list = [
                        metric(
                            all_labels.cpu(),
                            all_outputs.detach().cpu(),
                            **self.metrics[metric],
                        )
                        for metric in self.metrics
                    ]
                    metric_name_list = [
                        metric["type"] for metric in self._config.main_config.metrics
                    ]

                    train_scores = self.log(
                        train_loss / (step + 1),
                        train_loss_name,
                        metric_list,
                        metric_name_list,
                        train_logger,
                        train_log_values,
                        global_step,
                        append_text=self.train_config.append_text,
                    )
            pbar.close()
            if not os.path.exists(self.train_config.checkpoint.checkpoint_dir):
                os.makedirs(self.train_config.checkpoint.checkpoint_dir)

            if self.train_config.save_after_epoch:
                store_dict = {
                    "model_state_dict": model.state_dict(),
                }

                path = f"{self.train_config.checkpoint.checkpoint_dir}_{str(self.train_config.log.log_label)}_{str(epoch)}.pth"

                self.save(store_dict, path, save_flag=1)

        if epoch == max_epochs:
            # print("\nEvaluating\n")
            val_scores = self.val(
                model,
                val_dataset,
                criterion,
                device,
                global_step,
                train_logger,
                train_log_values,
            )

            # print("\nLogging\n")
            train_loss_name = self.train_config.criterion.type
            metric_list = [
                metric(
                    all_labels.cpu(), all_outputs.detach().cpu(), **self.metrics[metric]
                )
                for metric in self.metrics
            ]
            metric_name_list = [
                metric["type"] for metric in self._config.main_config.metrics
            ]

            train_scores = self.log(
                train_loss / len(train_loader),
                train_loss_name,
                metric_list,
                metric_name_list,
                train_logger,
                train_log_values,
                global_step,
                append_text=self.train_config.append_text,
            )

            if self.train_config.save_on is not None:

                ## BEST SCORES UPDATING

                train_scores = self.get_scores(
                    train_loss,
                    len(train_loader),
                    self.train_config.criterion.type,
                    all_outputs,
                    all_labels,
                )

                best_score, best_step, save_flag = self.check_best(
                    val_scores, save_on_score, best_score, global_step
                )

                store_dict = {
                    "model_state_dict": model.state_dict(),
                    "best_step": best_step,
                    "best_score": best_score,
                    "save_on_score": save_on_score,
                }

                path = self.train_config.save_on.best_path.format(self.log_label)

                self.save(store_dict, path, save_flag)

                if save_flag and train_log_values["hparams"] is not None:
                    (
                        best_hparam_list,
                        best_hparam_name_list,
                        best_metrics_list,
                        best_metrics_name_list,
                    ) = self.update_hparams(train_scores, val_scores, desc="best_val")

                ## FINAL SCORES UPDATING + STORING
                train_scores = self.get_scores(
                    train_loss,
                    len(train_loader),
                    self.train_config.criterion.type,
                    all_outputs,
                    all_labels,
                )

                store_dict = {
                    "model_state_dict": model.state_dict(),
                    "final_step": global_step,
                    "final_score": train_scores[save_on_score],
                    "save_on_score": save_on_score,
                }

                path = self.train_config.save_on.final_path.format(self.log_label)

                self.save(store_dict, path, save_flag=1)
                if train_log_values["hparams"] is not None:
                    (
                        final_hparam_list,
                        final_hparam_name_list,
                        final_metrics_list,
                        final_metrics_name_list,
                    ) = self.update_hparams(train_scores, val_scores, desc="final")
                    train_logger.save_hyperparams(
                        best_hparam_list,
                        best_hparam_name_list,
                        [int(self.log_label),] + best_metrics_list + final_metrics_list,
                        ["hparams/log_label",]
                        + best_metrics_name_list
                        + final_metrics_name_list,
                    )
                    #

    ## Need to check if we want same loggers of different loggers for train and eval
    ## Evaluate

    def get_scores(self, loss, divisor, loss_name, all_outputs, all_labels):

        avg_loss = loss / divisor

        metric_list = [
            metric(all_labels.cpu(), all_outputs.detach().cpu(), **self.metrics[metric])
            for metric in self.metrics
        ]
        metric_name_list = [
            metric["type"] for metric in self._config.main_config.metrics
        ]

        return dict(zip([loss_name,] + metric_name_list, [avg_loss,] + metric_list,))

    def check_best(self, val_scores, save_on_score, best_score, global_step):
        save_flag = 0
        best_step = global_step
        if self.train_config.save_on.desired == "min":
            if val_scores[save_on_score] < best_score:
                save_flag = 1
                best_score = val_scores[save_on_score]
                best_step = global_step
        else:
            if val_scores[save_on_score] > best_score:
                save_flag = 1
                best_score = val_scores[save_on_score]
                best_step = global_step
        return best_score, best_step, save_flag

    def update_hparams(self, train_scores, val_scores, desc):
        hparam_list = []
        hparam_name_list = []
        for hparam in self.train_config.log.values.hparams:
            hparam_list.append(get_item_in_config(self._config, hparam["path"]))
            if isinstance(hparam_list[-1], Config):
                hparam_list[-1] = hparam_list[-1].as_dict()
            hparam_name_list.append(hparam["name"])

        val_keys, val_values = zip(*val_scores.items())
        train_keys, train_values = zip(*train_scores.items())
        val_keys = list(val_keys)
        train_keys = list(train_keys)
        val_values = list(val_values)
        train_values = list(train_values)
        for i, key in enumerate(val_keys):
            val_keys[i] = f"hparams/{desc}_val_" + val_keys[i]
        for i, key in enumerate(train_keys):
            train_keys[i] = f"hparams/{desc}_train_" + train_keys[i]
        # train_logger.save_hyperparams(hparam_list, hparam_name_list,train_values+val_values,train_keys+val_keys, )
        return (
            hparam_list,
            hparam_name_list,
            train_values + val_values,
            train_keys + val_keys,
        )

    def save(self, store_dict, path, save_flag=0):
        if save_flag:
            dirs = "/".join(path.split("/")[:-1])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            torch.save(store_dict, path)

    def log(
        self,
        loss,
        loss_name,
        metric_list,
        metric_name_list,
        logger,
        log_values,
        global_step,
        append_text,
    ):

        return_dic = dict(zip([loss_name,] + metric_name_list, [loss,] + metric_list,))

        loss_name = f"{append_text}_{self.log_label}_{loss_name}"
        if log_values["loss"]:
            logger.save_params(
                [loss],
                [loss_name],
                combine=True,
                combine_name="losses",
                global_step=global_step,
            )

        for i in range(len(metric_name_list)):
            metric_name_list[
                i
            ] = f"{append_text}_{self.log_label}_{metric_name_list[i]}"
        if log_values["metrics"]:
            logger.save_params(
                metric_list,
                metric_name_list,
                combine=True,
                combine_name="metrics",
                global_step=global_step,
            )
            # print(hparams_list)
            # print(hparam_name_list)

        # for k,v in dict(zip([loss_name],[loss])).items():
        #     print(f"{k}:{v}")
        # for k,v in dict(zip(metric_name_list,metric_list)).items():
        #     print(f"{k}:{v}")
        return return_dic

    def val(
        self,
        model,
        dataset,
        criterion,
        device,
        global_step,
        train_logger=None,
        train_log_values=None,
        log=True,
    ):
        append_text = self.val_config.append_text
        if train_logger is not None:
            val_logger = train_logger
        else:
            val_logger = Logger(**self.val_config.log.logger_params.as_dict())

        if train_log_values is not None:
            val_log_values = train_log_values
        else:
            val_log_values = self.val_config.log.values.as_dict()
        if "custom_collate_fn" in dir(dataset):
            val_loader = DataLoader(
                dataset=dataset,
                collate_fn=dataset.custom_collate_fn,
                **self.val_config.loader_params.as_dict(),
            )
        else:
            val_loader = DataLoader(
                dataset=dataset, **self.val_config.loader_params.as_dict()
            )

        all_outputs = torch.Tensor().to(device)
        if self.train_config.label_type == "float":
            all_labels = torch.FloatTensor().to(device)
        else:
            all_labels = torch.LongTensor().to(device)

        batch_size = self.val_config.loader_params.batch_size

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for j, batch in enumerate(val_loader):

                inputs, labels = batch

                if self.train_config.label_type == "float":
                    labels = labels.float()

                for key in inputs:
                    inputs[key] = inputs[key].to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(torch.squeeze(outputs, dim=1), labels)
                val_loss += loss.item()

                all_labels = torch.cat((all_labels, labels), 0)

                if self.train_config.label_type == "float":
                    all_outputs = torch.cat((all_outputs, outputs), 0)
                else:
                    all_outputs = torch.cat(
                        (all_outputs, torch.argmax(outputs, axis=1)), 0
                    )

            val_loss = val_loss / len(val_loader)

            val_loss_name = self.train_config.criterion.type

            # print(all_outputs, all_labels)
            metric_list = [
                metric(
                    all_labels.cpu(), all_outputs.detach().cpu(), **self.metrics[metric]
                )
                for metric in self.metrics
            ]
            metric_name_list = [
                metric["type"] for metric in self._config.main_config.metrics
            ]
            return_dic = dict(
                zip([val_loss_name,] + metric_name_list, [val_loss,] + metric_list,)
            )
            if log:
                val_scores = self.log(
                    val_loss,
                    val_loss_name,
                    metric_list,
                    metric_name_list,
                    val_logger,
                    val_log_values,
                    global_step,
                    append_text,
                )
                return val_scores
            return return_dic
