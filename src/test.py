import torchaudio
from speechbrain.pretrained import EncoderClassifier
language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
# Download Thai language sample from Omniglot and cvert to suitable form
signal = language_id.load_audio("test_ru.wav")
prediction =  language_id.classify_batch(signal)
print(prediction)

# The scores in the prediction[0] tensor can be interpreted as log-likelihoods that
# the given utterance belongs to the given language (i.e., the larger the better)
# The linear-scale likelihood can be retrieved using the following:
print(prediction[1].exp())
# The identified language ISO code is given in prediction[3]
print(prediction[3])
# Alternatively, use the utterance embedding extractor:
emb =  language_id.encode_batch(signal)
print(emb.shape)
