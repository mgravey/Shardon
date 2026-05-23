CURRENTLY_LOADED_MODEL_ALIAS = "currently-loaded"


def is_currently_loaded_model_alias(model_name: str) -> bool:
    return model_name == CURRENTLY_LOADED_MODEL_ALIAS
