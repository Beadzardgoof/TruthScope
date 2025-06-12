import cv2
import numpy as np
from typing import Union, Dict, List
import torch
from pathlib import Path
import logging
from models import load_model  # Import from models.py in the same directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoTruthAnalyzer:
    """
    A class to analyze videos and determine truth ratings based on visual deception indicators.
    """
    
    def __init__(self, model_path: Union[str, Path] = None):
        """
        Initialize the VideoTruthAnalyzer with optional custom model path.
        
        Args:
            model_path (Union[str, Path], optional): Path to the custom model weights.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model(model_path)
        logger.info(f"Initialized VideoTruthAnalyzer using device: {self.device}")

    def _initialize_model(self, model_path: Union[str, Path] = None) -> torch.nn.Module:
        """
        Initialize the deception detection model.
        
        Args:
            model_path (Union[str, Path], optional): Path to model weights.
            
        Returns:
            torch.nn.Module: Loaded model
        """
        try:
            model = load_model()  # Load default model from Models.py
            if model_path:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single frame for model input.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            torch.Tensor: Preprocessed frame tensor
        """
        # Resize frame to expected input size
        frame = cv2.resize(frame, (224, 224))
        
        # Convert to float and normalize
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return frame_tensor.to(self.device)

    def analyze_video(self, video_path: Union[str, Path], 
                     frame_sample_rate: int = 30) -> Dict[str, float]:
        """
        Analyze a video file and return truth ratings.
        
        Args:
            video_path (Union[str, Path]): Path to the video file
            frame_sample_rate (int): Number of frames to skip between analyses
            
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'truth_score': Overall truth score (0-1, higher means more truthful)
                - 'confidence': Confidence in the prediction (0-1)
                - 'deception_indicators': Number of deception indicators found
        """
        try:
            video_path = str(video_path)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            frame_scores = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process every nth frame based on frame_sample_rate
                if frame_count % frame_sample_rate == 0:
                    processed_frame = self._preprocess_frame(frame)
                    
                    with torch.no_grad():
                        frame_score = self.model(processed_frame)
                        frame_scores.append(frame_score.cpu().numpy())
                
                frame_count += 1
            
            cap.release()
            
            # Calculate final scores
            frame_scores = np.array(frame_scores)
            truth_score = float(np.mean(frame_scores))
            confidence = float(np.std(frame_scores))
            deception_indicators = len([s for s in frame_scores if s < 0.5])
            
            return {
                "truth_score": truth_score,
                "confidence": confidence,
                "deception_indicators": deception_indicators
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            raise

def analyze_video_file(video_path: Union[str, Path], 
                      model_path: Union[str, Path] = None) -> Dict[str, float]:
    """
    Convenience function to quickly analyze a video file.
    
    Args:
        video_path (Union[str, Path]): Path to the video file
        model_path (Union[str, Path], optional): Path to custom model weights
        
    Returns:
        Dict[str, float]: Analysis results including truth score and confidence
    """
    analyzer = VideoTruthAnalyzer(model_path)
    return analyzer.analyze_video(video_path)

if __name__ == "__main__":
    # Example usage
    video_file = "path/to/video.mp4"
    try:
        results = analyze_video_file(video_file)
        print(f"Analysis Results:")
        print(f"Truth Score: {results['truth_score']:.2f}")
        print(f"Confidence: {results['confidence']:.2f}")
        print(f"Deception Indicators Found: {results['deception_indicators']}")
    except Exception as e:
        print(f"Error analyzing video: {str(e)}") 