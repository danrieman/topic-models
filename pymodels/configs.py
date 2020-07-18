import re


ALLOWED_TOPICS = {10, 25, 50, 100}

FEATURE_REGEX = re.compile("^Topic\\_[0-9]{1,}")

FACEBOOK_QUERY = "is_fb==1"
TWITTER_QUERY = "is_fb==0"

DATA_CONFIG = {
    "with_domain": {
        "test": "test_domain_{k}.csv",
        "train": "training_domain_{k}.csv",
    },
    "without_domain": {
        "test": "test_no_{k}.csv",
        "train": "training_no_{k}.csv",
    },
}

MODEL_CONFIG = {
    "age": {
        "model_type": "regression",
        "filter_query": "",
        "feature_regex": FEATURE_REGEX,
    },
    "gender": {
        "model_type": "classification",
        "filter_query": "gender != 3",
        "feature_regex": FEATURE_REGEX,
    },
}
