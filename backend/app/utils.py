from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def preprocess(audio_batch):
    return processor(
        list(audio_batch),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )