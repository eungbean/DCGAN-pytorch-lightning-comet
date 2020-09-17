from pytorch_lightning.loggers.comet import CometLogger
import logging

level = logging.INFO
logging.basicConfig(level=level, format="%(asctime)s - [%(levelname)s] - %(message)s")


def get_cometLogger(_C):
    comet_logger = CometLogger(
        api_key         = _C.COMET.APIKEY,
        workspace       = _C.COMET.WORKSPACE,
        project_name    = _C.COMET.PROJECT_NAME,
        experiment_name = _C.EXP_TITLE,
        save_dir        = str(_C.OUTPUT.LOG_ROOT),
        log_code        = _C.COMET.LOG_CODE,
        log_graph       = _C.COMET.LOG_GRAPH,
        auto_weight_logging=_C.COMET.AUTO_LOG_WEIGHT,
        auto_param_logging=_C.COMET.AUTO_LOG_PARAM,
        auto_metric_logging=_C.COMET.AUTO_LOG_METRIC,
        auto_output_logging=_C.COMET.AUTO_LOG_OUTPUT,
        # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        # disabled        = _C.COMET.COMET_LOGGER_DISABLE,
    )
    return comet_logger
