import json
import shutil
from testing_layer.configuration import Config
from testing_layer.workflows.enums import SupportedModels

if __name__ == "__main__":
    with open("./config-imagenet.json", "r") as file:
        obj = json.load(file)
        models = [SupportedModels(model) for model in obj.get("models")]

        for tested_model in models:
            obj['model'] = tested_model.value
            print(f"Make archive for model {tested_model.value}")
            config = Config(obj)
            shutil.make_archive( f"{config.model}-{config.tag}", "zip", f"counted_outputs/{config.model_base_dir}")
