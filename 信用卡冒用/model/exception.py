class InvalidTrainTypeException(Exception):
    "Raised when the train type is not either train, grid_search, or cross_validation"
    pass

class PipelineNotExistException(Exception):
    "Raised when pipeline not exists"
    pass