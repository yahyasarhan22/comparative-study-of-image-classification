# Yahya Sarhan 1221858 - Final Optimized Version
import numpy as np
import os
from skimage import io, transform, feature, color, filters, measure
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def extract_advanced_features(image):
    """
    Extract comprehensive features from an image for better classification.
    """
    features = []

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = color.rgb2gray(image)
        color_image = image
    else:
        gray_image = image
        color_image = np.stack([image, image, image], axis=-1)

    # === STATISTICAL FEATURES ===
    features.extend([
        np.mean(gray_image),
        np.std(gray_image),
        np.min(gray_image),
        np.max(gray_image),
        np.median(gray_image),
        np.percentile(gray_image, 25),
        np.percentile(gray_image, 75),
        np.var(gray_image)
    ])

    # === HISTOGRAM FEATURES ===
    # Gray histogram
    hist_gray, _ = np.histogram(gray_image.flatten(), bins=32, range=(0, 1))
    hist_gray = hist_gray / np.sum(hist_gray)
    features.extend(hist_gray[:16])  # Take first 16 bins

    # Color histograms if available
    if len(color_image.shape) == 3:
        for channel in range(3):
            hist_color, _ = np.histogram(color_image[:, :, channel].flatten(), bins=16, range=(0, 1))
            hist_color = hist_color / np.sum(hist_color) if np.sum(hist_color) > 0 else hist_color
            features.extend(hist_color[:8])  # Take first 8 bins per channel
    else:
        features.extend([0] * 24)  # Placeholder for color features

    # === TEXTURE FEATURES ===
    # HOG features
    try:
        hog_features = feature.hog(gray_image,
                                   orientations=9,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   block_norm='L2-Hys')
        # Sample HOG features to avoid too high dimensionality
        step = max(1, len(hog_features) // 50)
        features.extend(hog_features[::step][:50])
    except:
        features.extend([0] * 50)

    # Local Binary Pattern
    try:
        lbp = feature.local_binary_pattern(gray_image, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=10)
        lbp_hist = lbp_hist / np.sum(lbp_hist) if np.sum(lbp_hist) > 0 else lbp_hist
        features.extend(lbp_hist)
    except:
        features.extend([0] * 10)

    # === SHAPE FEATURES ===
    # Edge detection
    try:
        edges = feature.canny(gray_image, sigma=1.0)
        edge_density = np.sum(edges) / edges.size
        features.append(edge_density)
    except:
        features.append(0)

    # Gradient magnitude
    try:
        grad_x = filters.sobel_h(gray_image)
        grad_y = filters.sobel_v(gray_image)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude)
        ])
    except:
        features.extend([0, 0])

    # Regional properties
    try:
        # Create binary image
        binary = gray_image > filters.threshold_otsu(gray_image)
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)

        if props:
            # Get properties of the largest region
            largest_region = max(props, key=lambda x: x.area)
            features.extend([
                largest_region.area / gray_image.size,  # Relative area
                largest_region.eccentricity,
                largest_region.solidity,
                largest_region.extent
            ])
        else:
            features.extend([0, 0, 0, 0])
    except:
        features.extend([0, 0, 0, 0])

    return np.array(features, dtype=np.float64)


def load_images_final(image_folder, image_size=(64, 64)):
    """
    Final optimized image loading with comprehensive feature extraction.
    """
    images = []
    labels = []
    classes = sorted(os.listdir(image_folder))

    print(f"Found classes: {classes}")

    for class_label in classes:
        class_folder = os.path.join(image_folder, class_label)

        if os.path.isdir(class_folder):
            print(f"Processing class: {class_label}")
            class_images = 0

            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)

                if os.path.isfile(image_path) and image_name.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    try:
                        # Load and preprocess image
                        image = io.imread(image_path)

                        # Handle different formats
                        if len(image.shape) == 3 and image.shape[2] == 4:
                            image = image[:, :, :3]

                        # Resize with better quality
                        image_resized = transform.resize(image, image_size,
                                                         anti_aliasing=True,
                                                         preserve_range=True)
                        image_resized = image_resized / 255.0

                        # Extract features
                        feature_vector = extract_advanced_features(image_resized)

                        # Clean features
                        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)

                        images.append(feature_vector)
                        labels.append(class_label)
                        class_images += 1

                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

            print(f"  Loaded {class_images} images from {class_label}")

    print(f"Total images loaded: {len(images)}")

    X = np.array(images, dtype=np.float64)
    y = np.array(labels, dtype=str)

    return X, y


def train_optimized_models(X_train, y_train, X_test, y_test):
    """
    Train multiple optimized models and return results.
    """
    models = {}
    results = {}

    # Encode labels for some models
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training optimized models...")

    # 1. Optimized Naive Bayes
    print("1. Training Naive Bayes...")
    nb_model = GaussianNB(var_smoothing=1e-8)
    nb_model.fit(X_train_scaled, y_train)
    nb_pred = nb_model.predict(X_test_scaled)
    nb_accuracy = accuracy_score(y_test, nb_pred)

    models['Naive Bayes'] = nb_model
    results['Naive Bayes'] = {
        'accuracy': nb_accuracy,
        'predictions': nb_pred,
        'model': nb_model
    }

    # 2. Optimized Decision Tree
    print("2. Training Decision Tree...")
    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        criterion='gini',
        max_features='sqrt',
        class_weight='balanced'
    )
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)

    models['Decision Tree'] = dt_model
    results['Decision Tree'] = {
        'accuracy': dt_accuracy,
        'predictions': dt_pred,
        'model': dt_model
    }

    # 3. Optimized Neural Network
    print("3. Training Neural Network...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(200, 150, 100),
        max_iter=300,
        random_state=42,
        alpha=0.0001,
        learning_rate_init=0.01,
        solver='adam',
        batch_size=32,
        beta_1=0.9,
        beta_2=0.999
    )
    nn_model.fit(X_train_scaled, y_train)
    nn_pred = nn_model.predict(X_test_scaled)
    nn_accuracy = accuracy_score(y_test, nn_pred)

    models['Neural Network'] = nn_model
    results['Neural Network'] = {
        'accuracy': nn_accuracy,
        'predictions': nn_pred,
        'model': nn_model
    }

    return results, scaler, le


def evaluate_final_results(results, y_test):
    """
    Comprehensive evaluation of all models.
    """
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    for i, (model_name, result) in enumerate(sorted_results, 1):
        accuracy = result['accuracy']
        print(f"{i}. {model_name}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\n" + "=" * 60)

    # Detailed analysis for each model
    for model_name, result in results.items():
        print(f"\n--- {model_name} Detailed Report ---")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, result['predictions']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, result['predictions'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {result["accuracy"]:.4f}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()

    # Comparison chart
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]

    plt.figure(figsize=(12, 8))
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = plt.bar(model_names, accuracies, color=colors[:len(model_names)])

    plt.title('Final Model Comparison - Accuracy Scores', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}\n({acc * 100:.1f}%)', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    return sorted_results[0]  # Return best model


# Main execution
if __name__ == "__main__":
    print("=== FINAL OPTIMIZED IMAGE CLASSIFICATION PROJECT ===")
    print("Advanced feature extraction + Optimized models")

    # Configuration
    IMAGE_FOLDER = r"C:\Users\yahya_k6rln48\OneDrive\Desktop\project2_AI\dataset1"
    IMAGE_SIZE = (64, 64)

    # Load dataset
    print("\n1. Loading dataset with advanced feature extraction...")
    X, y = load_images_final(IMAGE_FOLDER, IMAGE_SIZE)

    if len(X) == 0:
        print("No images loaded. Check your path.")
        exit()

    print(f"Dataset shape: {X.shape}")
    print(f"Feature vector length: {X.shape[1]}")
    print(f"Classes: {np.unique(y)}")

    # Split dataset
    print("\n2. Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train models
    print("\n3. Training optimized models...")
    results, scaler, le = train_optimized_models(X_train, y_train, X_test, y_test)

    # Evaluate results
    print("\n4. Evaluating results...")
    best_model_info = evaluate_final_results(results, y_test)

    print(f"\n🏆 BEST MODEL: {best_model_info[0]} with {best_model_info[1]['accuracy']:.4f} accuracy")