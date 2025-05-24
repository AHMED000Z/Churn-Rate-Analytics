import joblib
import warnings

print("Testing model loading...")

try:
    print("Attempting to load model...")
    model = joblib.load("best_model.pkl")
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
    
    # Check if it's a Random Forest model
    if hasattr(model, 'estimators_'):
        print(f"Number of trees: {len(model.estimators_)}")
    
    # Check expected features
    if hasattr(model, 'feature_names_in_'):
        print(f"Number of features: {len(model.feature_names_in_)}")
        print(f"Feature names: {model.feature_names_in_}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    
    print("Trying with warnings suppressed...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            model = joblib.load("best_model.pkl")
            print("Model loaded with warnings suppressed!")
        except Exception as e2:
            print(f"Failed even with warnings suppressed: {e2}")