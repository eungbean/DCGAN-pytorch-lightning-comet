from pytorch_lightning.loggers.comet import CometLogger
import logging

level = logging.INFO
logging.basicConfig(level=level, format='%(asctime)s - [%(levelname)s] - %(message)s')

def get_cometLogger(_C):
    comet_logger = CometLogger(
        api_key         = _C.COMET.APIKEY,
        workspace       = _C.COMET.WORKSPACE,
        project_name    = _C.COMET.PROJECT_NAME,
        experiment_name = _C.EXP_TITLE,
        save_dir        = str(_C.OUTPUT.LOG_ROOT),
        log_graph       = True,
        # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        # disabled        = _C.COMET.COMET_LOGGER_DISABLE,
        #auto_metric_logging = False,
    )
    return comet_logger