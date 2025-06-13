import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ExerciseDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, landmarks):
        """Extract relevant features from pose landmarks"""
        features = []
        
        # Extract key joint positions
        key_points = [
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]
        
        for point in key_points:
            if hasattr(landmarks, point):
                point_value = getattr(landmarks, point).value
                features.extend([
                    landmarks[point_value].x,
                    landmarks[point_value].y,
                    landmarks[point_value].z
                ])
        
        return np.array(features)
    
    def train(self, X, y):
        """Train the exercise classifier"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, landmarks):
        """Predict exercise type from landmarks"""
        if not self.is_trained:
            return None
            
        features = self.extract_features(landmarks)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict(features_scaled)[0]
    
    def save_model(self, model_path='exercise_model.joblib'):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, model_path)
    
    def load_model(self, model_path='exercise_model.joblib'):
        """Load a trained model"""
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = True
            return True
        return False 