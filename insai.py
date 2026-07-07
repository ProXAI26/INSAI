#pip install kneed
#pip install tensorflow
#pip install opencv-python
#pip install scikit-learn
#pip install matplotlib
#pip install pandas
#pip install scikit-image

import os
import gc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import shutil
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import re
import subprocess
import tensorflow.keras.backend as K
import random
from sklearn.utils import shuffle

"""# **Data Loading**"""

from concurrent.futures import ThreadPoolExecutor


dataset_path = 'D:\INSAI\Images'
target_size = (224, 224)

image_paths = []
labels = []


for class_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        image_paths += [os.path.join(folder_path, f) for f in files]
        labels += [class_folder] * len(files)


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)


train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, categorical_labels,
    test_size=0.2, random_state=42, stratify=categorical_labels
)


def load_and_preprocess(path):
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, target_size)
        return img.astype("float32") / 255.0
    return None

def load_images_parallel(paths, max_workers=16):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(load_and_preprocess, paths))
    return np.array([img for img in images if img is not None], dtype="float32")


train_images = load_images_parallel(train_paths)
test_images = load_images_parallel(test_paths)


print(f"Total images: {len(image_paths)}")
print(f"Training set: {len(train_images)} images")
print(f"Test set: {len(test_images)} images")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Image shape: {train_images[0].shape}")

"""# **Loss Function**"""

def custom_combo_loss(model,alpha,false_count):
    def loss_fn(y_true, y_pred):

        cnn_loss = K.categorical_crossentropy(y_true, y_pred)

        false_count1 = false_count/420

        reg_loss = tf.add_n([
            tf.reduce_mean(tf.abs(w))
            for w in model.trainable_weights])

        return (1 - alpha) * cnn_loss + alpha * false_count1*reg_loss

    return loss_fn

"""# **CNN**"""

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[-3:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(21, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

optimizer = SGD(learning_rate=0.0001, momentum=0.9)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6)
model.summary()

"""# **GRAD CAM++**"""

def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        target_class_score = preds[:, pred_index]

    grads = tape.gradient(target_class_score, last_conv_layer_output)
    grads_squared = tf.square(grads)
    #grads_third = grads_squared * grads

    sum_grads = tf.reduce_sum(grads, axis=(0, 1, 2))
    sum_grads_squared = tf.reduce_sum(grads_squared, axis=(0, 1, 2))
    #sum_grads_third = tf.reduce_sum(grads_third, axis=(0, 1, 2))

    alpha = sum_grads / (sum_grads_squared + 1e-10)
    weighted_grads = alpha * grads
    pooled_grads = tf.reduce_mean(weighted_grads, axis=(0, 1, 2))

    weighted_feature_map = last_conv_layer_output[0] * pooled_grads
    heatmap = tf.reduce_sum(weighted_feature_map, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) > 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()

def extract_patches_from_heatmap(original_image, heatmap, top_k_percent=20, min_patch_size=100, target_size=(224, 224)):
    original_image = cv2.resize(original_image, (256, 256))
    heatmap_resized = cv2.resize(heatmap, (256, 256))


    threshold = np.percentile(heatmap_resized, 100 - top_k_percent)
    mask = np.uint8((heatmap_resized >= threshold) * 255)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    patch_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_patch_size:
            continue

        patch = original_image[y:y+h, x:x+w]
        if patch.size == 0:
            continue
        patch_resized = cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA)
        patch_images.append(patch_resized)

    return patch_images

def patch_images_creation_from_train_memory(images, labels, label_encoder, model_path, last_conv_layer_name):
    from tensorflow.keras.models import load_model
    model = load_model(model_path, custom_objects={'loss_fn': custom_combo_loss(0.5, false_count)})
    images, labels = shuffle(images, labels, random_state=42)

    all_patches, patch_labels, patch_names = [], [], []
    for class_index in range(len(label_encoder.classes_)):
        indices = [i for i, y in enumerate(labels) if np.argmax(y) == class_index]
        if not indices:
            continue
        idx = random.choice(indices)
        image = images[idx]
        original_img = (image * 255).astype(np.uint8)

        heatmap = make_gradcam_plus_plus_heatmap(
            img_array=np.expand_dims(image, axis=0),
            model=model,
            last_conv_layer_name=last_conv_layer_name,
            pred_index=class_index
        )

        patch_list = extract_patches_from_heatmap(original_img, heatmap, top_k_percent=20, min_patch_size=100, target_size=(224, 224))
        for i, patch in enumerate(patch_list):
            patch_filename = f"class{class_index}_patch{i}.png"
            all_patches.append(patch)
            patch_labels.append(class_index)
            patch_names.append(patch_filename)

    return all_patches, patch_labels, patch_names

def patch_images_creation_from_test_memory(images, labels, label_encoder, model_path, last_conv_layer_name):
    from tensorflow.keras.models import load_model
    model = load_model(model_path, custom_objects={'loss_fn': custom_combo_loss(0.5, false_count)})
    images, labels = shuffle(images, labels, random_state=42)

    all_patches, patch_labels, patch_names = [], [], []
    for class_index in range(len(label_encoder.classes_)):
        indices = [i for i, y in enumerate(labels) if np.argmax(y) == class_index]
        selected_indices = random.sample(indices, min(100, len(indices)))

        for idx in selected_indices:
            image = images[idx]
            original_img = (image * 255).astype(np.uint8)

            heatmap = make_gradcam_plus_plus_heatmap(
                img_array=np.expand_dims(image, axis=0),
                model=model,
                last_conv_layer_name=last_conv_layer_name,
                pred_index=class_index
            )

            patch_list = extract_patches_from_heatmap(original_img, heatmap, top_k_percent=20, min_patch_size=10, target_size=(224, 224))
            for i, patch in enumerate(patch_list):
                patch_filename = f"class{class_index}_img{idx}_patch{i}.png"
                all_patches.append(patch)
                patch_labels.append(class_index)
                patch_names.append(patch_filename)

    return all_patches, patch_labels, patch_names

"""# **Centroid Creation**"""

def centroid_clustering_from_array(patch_images, patch_class_names, patch_names):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    feature_vectors = [img.flatten() for img in patch_images]
    feature_vectors = StandardScaler().fit_transform(feature_vectors)

    pca = PCA(n_components=min(50, len(feature_vectors)))
    reduced = pca.fit_transform(feature_vectors)

    best_k, best_score, best_labels, best_kmeans = None, -1, None, None
    for k in range(2, min(len(reduced), 80)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced)
        score = silhouette_score(reduced, labels)
        if score > best_score:
            best_k, best_score, best_labels, best_kmeans = k, score, labels, kmeans

    centroids_dict = {}
    for i, centroid in enumerate(best_kmeans.cluster_centers_):
        closest_idx = np.argmin(np.linalg.norm(reduced - centroid, axis=1))
        centroids_dict[f"Cluster_{i}"] = patch_images[closest_idx]

    return centroids_dict

"""# **Rule Creation through checking with the centroid using Euclidian Distance**"""

def rule_creation_to_list_from_array(patch_images, patch_names, centroids):
    from scipy.spatial.distance import euclidean
    from collections import defaultdict

    image_clusters = defaultdict(set)
    for patch, name in zip(patch_images, patch_names):
        base_name = "_".join(name.split("_")[:-1])
        patch_flat = patch.flatten()

        best_cluster = None
        min_dist = float('inf')
        for cluster_id, centroid_img in centroids.items():
            centroid_flat = centroid_img.flatten()
            if centroid_flat.shape == patch_flat.shape:
                dist = euclidean(patch_flat, centroid_flat)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_id

        if best_cluster:
            image_clusters[base_name].add(best_cluster)

    rules = []
    for base_name, cluster_set in image_clusters.items():
        class_name = base_name.split("_")[0]
        cluster_str = ",".join(sorted(cluster_set))
        rule = f"Class: {class_name} | Image: {base_name} belongs to cluster(s): {cluster_str}"
        rules.append(rule)

    return rules

"""# **Rule Based AI Model Creation through Experta Library**"""

##use this one


import os

subprocess.run(["pip", "install", "experta"])
subprocess.run(["pip", "install", "--upgrade", "frozendict"])

from experta import *

def generate_experta_script_from_array(rules_list, output_folder):
    rules = []
    rule_counter = 1
    class_rules = {}

    for line in rules_list:
        line = line.strip()
        if not line:
            continue


        match = re.match(r"Class: (.+?) \| Image: .+? belongs to cluster\(s\): (.+)", line)
        if not match:
            continue

        class_name, cluster_str = match.groups()
        clusters = [f"Cluster_{c}" if not c.startswith("Cluster_") else c for c in cluster_str.split(",")]

        clusters_tuple = tuple(sorted(clusters))
        if class_name not in class_rules:
            class_rules[class_name] = set()

        if clusters_tuple in class_rules[class_name]:
            continue

        class_rules[class_name].add(clusters_tuple)

        if len(clusters) > 1:
            rule_type = "AND"
            salience = 3
        else:
            rule_type = "OR"
            salience = 1

        cluster_conditions = ",\n        ".join([f"Cluster(name='{cluster}')" for cluster in clusters])

        rule = f"""
    @Rule({rule_type}(
        {cluster_conditions}
    ), salience={salience})
    def classify_{class_name}_{rule_counter}(self):
        self.matched_classes.add("{class_name}")
        """
        rules.append(rule)
        rule_counter += 1


    script_content = f"""
from experta import *

class Cluster(Fact):
    pass

class ClassificationEngine(KnowledgeEngine):

    def __init__(self):
        super().__init__()
        self.matched_classes = set()

    def print_results(self):
        if self.matched_classes:
            print("Class:", ", ".join(sorted(self.matched_classes)))

    {''.join(rules)}

def run_engine(dataset):
    engine = ClassificationEngine()
    engine.reset()
    for data in dataset:
        for cluster in data:
            engine.declare(Cluster(name=cluster))
        engine.run()
    engine.print_results()
    return engine.matched_classes if engine.matched_classes else set()

if __name__ == '__main__':
    sample_dataset = [['Cluster_1'], ['Cluster_2', 'Cluster_3'], ['Cluster_4']]
    run_engine(sample_dataset)
"""


    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "classification_engine_from_array.py")

    with open(output_file, 'w') as file:
        file.write(script_content)

    print(f"Python script saved to: {output_file}")

"""# **False Count**"""

import importlib.util

def clean_name(name):

    return re.sub(r'[^a-zA-Z]', '', name).lower()

def parse_rule_line(line):

    match = re.match(r'Class:\s*(.+?)\s*\|\s*Image:\s*(.+?)\s*belongs to cluster\(s\):\s*(.+)', line)
    if not match:
        return None, None, None
    class_name, image_name, clusters = match.groups()
    cluster_list = [c.strip() for c in clusters.split(',')]
    return class_name, image_name, cluster_list

def false_count_creation_from_array(rule_list, experta_script_path):
    spec = importlib.util.spec_from_file_location(
        "classification_engine_from_array",
        os.path.join(experta_script_path, "classification_engine_from_array.py")
    )
    engine_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine_module)
    run_engine = engine_module.run_engine

    false_count = 0

    for rule_line in rule_list:
        parsed = parse_rule_line(rule_line)
        if parsed is None:
            continue

        true_class, image_name, cluster_list = parsed

        matched_classes = run_engine([cluster_list]) or set()

        cleaned_true = clean_name(true_class)
        cleaned_predicted = {clean_name(cls) for cls in matched_classes}

        is_match = any(
            cleaned_true in pred or pred in cleaned_true
            for pred in cleaned_predicted
        )

        if not is_match:
            false_count += 1

    print(false_count)
    return false_count

"""# **Number of Epoch**"""

i=int(input("Enter the number of epochs: "))
print(i)

"""# **Final Flow**"""

import os
import time

false_count = 0
result_path = "D:\INSAI\Result_final.txt"
start_time = time.time()
for x in range(i):
    print(f"\n=== Starting Iteration {x + 1} ===")
    # === [1] MODEL TRAINING ===
    model.compile(optimizer=optimizer, loss=custom_combo_loss(0.5, false_count), metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=1, batch_size=32, validation_data=(test_images, test_labels))
    train_accuracy = history.history['accuracy'][-1]
    test_accuracy = history.history['val_accuracy'][-1]

    # === [2] SAVE MODEL ===
    model_path = "D:\INSAI\My_model\MODEL2.h5"
    model.save(model_path)

    # === [3] TRAIN PATCHES (1 image/class) ===
    train_patches, train_patch_labels, train_patch_names = patch_images_creation_from_train_memory(
        train_images, train_labels, label_encoder, model_path, last_conv_layer_name="conv5_block3_out"
    )

    # === [4] TEST PATCHES (50 images/class) ===
    test_patches, test_patch_labels, test_patch_names = patch_images_creation_from_test_memory(
        test_images, test_labels, label_encoder, model_path, last_conv_layer_name="conv5_block3_out"
    )

    # === [5] CLUSTERING + CENTROID CREATION (In-memory) ===
    centroids_dict = centroid_clustering_from_array(
        patch_images=train_patches,
        patch_class_names=train_patch_labels,
        patch_names=train_patch_names
    )

    # === [6] RULE CREATION FROM TRAIN + TEST PATCHES ===
    rules_list_train = rule_creation_to_list_from_array(
        patch_images=train_patches,
        patch_names=train_patch_names,
        centroids=centroids_dict
    )

    rules_list_test = rule_creation_to_list_from_array(
        patch_images=test_patches,
        patch_names=test_patch_names,
        centroids=centroids_dict
    )

    # === [7] EXPERTA SCRIPT GENERATION ===
    experta_folder = "D:\INSAI\experta"
    generate_experta_script_from_array(rules_list_train, experta_folder)

    # === [8] FALSE COUNT CALCULATION ===
    false_count = false_count_creation_from_array(rules_list_test, experta_folder)

    # === [9] LOG ACCURACY + FALSE COUNT ===
    with open(result_path, 'a') as f:
        f.write(f"Iteration {x + 1} - False Count: {false_count}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}\n")

    # === [10] FINAL ITERATION SAVING (Clusters + Patches) ===
    if x == i - 1:
        # -- Save centroids (1 per cluster)
        cluster_save_path = "D:\INSAI\cluster1"
        os.makedirs(cluster_save_path, exist_ok=True)
        for cluster_id, img in centroids_dict.items():
            cluster_dir = os.path.join(cluster_save_path, cluster_id)
            os.makedirs(cluster_dir, exist_ok=True)
            cv2.imwrite(os.path.join(cluster_dir, f"centroid_{cluster_id}.png"), img)

        # -- Save train patches
        patch_train_dir = "D:\INSAI\patch"
        os.makedirs(patch_train_dir, exist_ok=True)
        for patch, name in zip(train_patches, train_patch_names):
            cv2.imwrite(os.path.join(patch_train_dir, name), patch)

        # -- Save test patches
        patch_test_dir = "D:\INSAI\Test_patch"
        os.makedirs(patch_test_dir, exist_ok=True)
        for patch, name in zip(test_patches, test_patch_names):
            cv2.imwrite(os.path.join(patch_test_dir, name), patch)

    # === [11] CLEANUP RAM + TEMP FOLDERS ===
    del train_patches, train_patch_labels, train_patch_names
    del test_patches, test_patch_labels, test_patch_names
    del centroids_dict, rules_list_train, rules_list_test
    gc.collect()

    if os.path.exists(experta_folder):
        shutil.rmtree(experta_folder, ignore_errors=True)
    elapsed_time = time.time() - start_time
    print(f"=== Iteration {x + 1} completed in {elapsed_time:.2f} seconds ===")




