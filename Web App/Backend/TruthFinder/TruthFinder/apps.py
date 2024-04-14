from django.apps import AppConfig
from keras.models import load_model

class TruthFinderConfig(AppConfig):
    name = 'TurthFinder'
    verbose_name = "Lie Detection"

    # Path to the models
    video_model_path = 'api/Pretrained Models/Video CNN_3D 0.61 Lss 87.50 Acc.h5'
    audio_model_path = 'api/Pretrained Models/Audio Bidirectional LSTM 0.62 Lss 79.17 Acc.h5'
    text_model_path = 'api/Pretrained Models/Text ANN tf-idf 0.50 Lss 79.17 Acc.h5'

    # Initialize models as class variables
    video_model = None
    audio_model = None
    text_model = None

    def ready(self):
        # Load models
        TruthFinderConfig.video_model = load_model(TruthFinderConfig.video_model_path)
        TruthFinderConfig.audio_model = load_model(TruthFinderConfig.audio_model_path)
        TruthFinderConfig.text_model = load_model(TruthFinderConfig.text_model_path)
