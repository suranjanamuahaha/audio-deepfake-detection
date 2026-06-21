from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "microsoft/wavlm-base-plus"
)

def preprocess(audio_batch):

    return feature_extractor(
        list(audio_batch),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )