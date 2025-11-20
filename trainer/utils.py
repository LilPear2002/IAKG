from logging import getLogger


class DisabledSummaryWriter:
    """Disabled TensorBoard writer for simplicity"""
    
    def __init__(*args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return self
    
    def __getattr__(self, *args, **kwargs):
        return self


def log_exceptions(func):
    """Decorator to log exceptions during training"""
    
    def wrapper(*args, **kwargs):
        logger = getLogger('train_logger')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e
    return wrapper
