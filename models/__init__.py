import importlib


def find_model_using_name(model_name):
    # the file "models/model_{model_name}.py" will be imported
    model_filename = "models.model_{}".format(model_name)
    modellib = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace("_", "") + "model"
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In {}.py, not find class [%s].".format(model_filename, target_model_name))
        exit(0)

    return model
