import os
import time
import base64
import joblib
import librosa
import tempfile
import numpy as np
import streamlit as st
import sounddevice as sd
from collections import Counter
import numpy as np
from collections import Counter
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from scipy.io.wavfile import write, read

SAMPLE_RATE = 44100
DURATION = 3

@dataclass
class TreeNode:
    """Node structure for decision tree"""
    feature_idx: int = None
    threshold: float = None
    left: any = None
    right: any = None
    value: any = None
    gain: float = None

class DecisionTree:
    """
    Decision Tree Classifier implementation
    """
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 1,
        min_information_gain: float = 0.0
    ) -> None:
        """
        Initialize Decision Tree with hyperparameters

        Args:
            max_depth: Maximum depth of the tree
            min_samples_leaf: Minimum samples required at leaf node
            min_information_gain: Minimum information gain required for split
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.root = None
        self.n_classes = None

    def entropy(self, class_probabilities: List[float]) -> float:
        """
        Calculate entropy for given probability distribution

        Args:
            class_probabilities: List of class probabilities

        Returns:
            float: Entropy value
        """
        return -np.sum([p * np.log2(p) for p in class_probabilities if p > 0])

    def class_probabilities(self, labels: List) -> List[float]:
        """
        Calculate class probabilities from labels

        Args:
            labels: List of class labels

        Returns:
            List[float]: Probability distribution over classes
        """
        total_count = len(labels)
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / total_count
        return probabilities

    def data_entropy(self, labels: List) -> float:
        """
        Calculate entropy of a dataset

        Args:
            labels: List of class labels

        Returns:
            float: Entropy of dataset
        """
        probabilities = self.class_probabilities(labels)
        return self.entropy(probabilities)

    def partition_entropy(self, subsets: List) -> float:
        """
        Calculate entropy of a partition

        Args:
            subsets: List of subsets (left and right splits)

        Returns:
            float: Weighted average entropy of subsets
        """
        total_count = sum(len(subset) for subset in subsets)
        weights = [len(subset) / total_count for subset in subsets]
        entropies = [self.data_entropy(subset[:, -1]) for subset in subsets]
        return sum(w * e for w, e in zip(weights, entropies))

    def split(self, data: np.array, feature_idx: int, threshold: float) -> Tuple:
        """
        Split dataset based on feature and threshold

        Args:
            data: Input data
            feature_idx: Index of feature to split on
            threshold: Threshold value for split

        Returns:
            Tuple: Left and right split datasets
        """
        left_mask = data[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return data[left_mask], data[right_mask]

    def find_best_split(self, data: np.array) -> Tuple:
        """
        Find best split for dataset
        Returns:
            Tuple: Best feature index, threshold, and information gain
        """
        n_features = data.shape[1] - 1
        parent_entropy = self.data_entropy(data[:, -1])
        best_gain = -1
        best_split = None

        for feature_idx in range(n_features):
            thresholds = np.unique(data[:, feature_idx])

            for threshold in thresholds:
                left_data, right_data = self.split(data, feature_idx, threshold)

                # Check minimum samples constraint
                if (len(left_data) < self.min_samples_leaf or
                    len(right_data) < self.min_samples_leaf):
                    continue

                # Calculate information gain
                split_entropy = self.partition_entropy([left_data, right_data])
                information_gain = parent_entropy - split_entropy

                # Update best split if this split is better
                if information_gain > best_gain and information_gain > self.min_information_gain:
                    best_gain = information_gain
                    best_split = (feature_idx, threshold, information_gain)

        return best_split

    def find_label_probs(self, data: np.array) -> np.array:
        """
        Calculate probability distribution over classes for a node
        Returns:
            np.array: Probability distribution over classes
        """
        label_counts = np.bincount(data[:, -1].astype(int), minlength=self.n_classes)
        return label_counts / len(data)

    def create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        """
        Recursively create decision tree

        Args:
            data: Input data
            current_depth: Current depth in tree

        Returns:
            TreeNode: Root node of (sub)tree
        """
        node = TreeNode()

        # Check stopping criteria
        if (current_depth >= self.max_depth or
            len(data) < 2 * self.min_samples_leaf or
            len(np.unique(data[:, -1])) == 1):
            node.value = self.find_label_probs(data)
            return node

        # Find best split
        best_split = self.find_best_split(data)

        # If no valid split found, make leaf node
        if best_split is None:
            node.value = self.find_label_probs(data)
            return node

        # Create split node
        feature_idx, threshold, gain = best_split
        left_data, right_data = self.split(data, feature_idx, threshold)

        node.feature_idx = feature_idx
        node.threshold = threshold
        node.gain = gain
        node.left = self.create_tree(left_data, current_depth + 1)
        node.right = self.create_tree(right_data, current_depth + 1)

        return node

    def predict_one_sample(self, x: np.array) -> np.array:
        """
        Predict class probabilities for single sample
        """
        node = self.root

        while node.value is None:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predict class probabilities for multiple samples
        """
        return np.array([self.predict_one_sample(x) for x in X])

    def predict(self, X: np.array) -> np.array:
        """
        Predict class labels for multiple samples
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def train(self, X_train: np.array, y_train: np.array) -> None:
        """
        Train decision tree

        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Get number of classes
        self.n_classes = len(np.unique(y_train))

        # Combine features and labels
        data = np.column_stack([X_train, y_train])

        # Create tree
        self.root = self.create_tree(data, current_depth=0)

    def print_recursive(self, node: TreeNode, level: int = 0) -> None:
        """
        Print tree structure recursively

        Args:
            node: Current node
            level: Current level in tree
        """
        indent = "  " * level

        if node.value is not None:
            print(f"{indent}Leaf: class probabilities = {node.value}")
            return

        # Change here: convert node.threshold to a single float for formatting
        threshold_value = node.threshold.item() if isinstance(node.threshold, np.ndarray) else node.threshold
        print(f"{indent}Split: feature {node.feature_idx}, threshold = {threshold_value:.4f}, gain = {node.gain:.4f}")
        print(f"{indent}Left:")
        self.print_recursive(node.left, level + 1)
        print(f"{indent}Right:")
        self.print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        """Print entire tree structure"""
        print("Decision Tree Structure:")
        self.print_recursive(self.root)

# Fungsi untuk menghitung entropi
def hitung_entropi(y):
    label, jumlah = np.unique(y, return_counts=True)
    proporsi = jumlah / len(y)
    return -np.sum(proporsi * np.log2(proporsi))

# Fungsi untuk mempartisi dataset
def partisi(data, fitur, nilai):
    data_kiri = data[data[:, fitur] <= nilai]
    data_kanan = data[data[:, fitur] > nilai]
    return data_kiri, data_kanan

# Fungsi untuk menemukan split terbaik
def cari_split_terbaik(data):
    n_fitur = data.shape[1] - 1  # Kolom terakhir adalah label
    entropi_terbaik = float("inf")
    split_terbaik = None

    for fitur in range(n_fitur):
        nilai_unik = np.unique(data[:, fitur])
        for nilai in nilai_unik:
            data_kiri, data_kanan = partisi(data, fitur, nilai)
            if len(data_kiri) > 0 and len(data_kanan) > 0:
                proporsi_kiri = len(data_kiri) / len(data)
                proporsi_kanan = len(data_kanan) / len(data)
                entropi_split = (proporsi_kiri * hitung_entropi(data_kiri[:, -1]) +
                                 proporsi_kanan * hitung_entropi(data_kanan[:, -1]))
                if entropi_split < entropi_terbaik:
                    entropi_terbaik = entropi_split
                    split_terbaik = {
                        "fitur": fitur,
                        "nilai": nilai,
                        "data_kiri": data_kiri,
                        "data_kanan": data_kanan
                    }
    return split_terbaik

# Node Pohon Keputusan
class Node:
    def __init__(self, fitur=None, nilai=None, label=None, data_kiri=None, data_kanan=None):
        self.fitur = fitur
        self.nilai = nilai
        self.label = label
        self.data_kiri = data_kiri
        self.data_kanan = data_kanan

# Fungsi untuk membangun pohon keputusan
def bangun_pohon(data, max_kedalaman, kedalaman=0):
    label, jumlah = np.unique(data[:, -1], return_counts=True)
    if len(label) == 1 or kedalaman == max_kedalaman:
        return Node(label=label[np.argmax(jumlah)])

    split = cari_split_terbaik(data)
    if not split:
        return Node(label=label[np.argmax(jumlah)])

    node = Node(fitur=split["fitur"], nilai=split["nilai"])
    node.data_kiri = bangun_pohon(split["data_kiri"], max_kedalaman, kedalaman + 1)
    node.data_kanan = bangun_pohon(split["data_kanan"], max_kedalaman, kedalaman + 1)
    return node

# Fungsi untuk prediksi
def prediksi_satu(data, pohon):
    if pohon.label is not None:
        return pohon.label
    if data[pohon.fitur] <= pohon.nilai:
        return prediksi_satu(data, pohon.data_kiri)
    else:
        return prediksi_satu(data, pohon.data_kanan)

# Fungsi Random Forest
class RandomForest:
    def __init__(self, n_pohon=10, max_kedalaman=5):
        self.n_pohon = n_pohon
        self.max_kedalaman = max_kedalaman
        self.hutan = []

    def pelatihan(self, data):
        n_sampel = data.shape[0]
        for _ in range(self.n_pohon):
            indeks_acak = np.random.choice(n_sampel, n_sampel, replace=True)
            data_bootstrap = data[indeks_acak]
            pohon = bangun_pohon(data_bootstrap, self.max_kedalaman)
            self.hutan.append(pohon)

    def prediksi(self, data):
        prediksi_semua = np.array([prediksi_satu(data, pohon) for pohon in self.hutan])
        return Counter(prediksi_semua).most_common(1)[0][0]

# Fungsi train-test split
def bagi_data(data, rasio_uji=0.2):
    np.random.shuffle(data)
    batas = int(len(data) * (1 - rasio_uji))
    data_latih = data[:batas]
    data_uji = data[batas:]
    return data_latih, data_uji

# Fungsi untuk menghitung akurasi
def hitung_akurasi(data_uji, model):
    benar = 0
    for sampel in data_uji:
        pred = model.prediksi(sampel[:-1])
        if pred == sampel[-1]:
            benar += 1
    return benar / len(data_uji)

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2)**2))

# Fungsi untuk prediksi KNN
def knn_predict(train_data, train_labels, test_row, k=1):
    distances = []
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_row)
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:k]]
    return Counter(k_nearest).most_common(1)[0][0]

def extract_features1(file_path):
    audio, src = librosa.load(file_path)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=src, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=src)
    mean_chroma = np.mean(chroma, axis=1)
    
    # Delta
    mfccs_delta = librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=src, n_mfcc=13))
    mfccs_delta = np.mean(mfccs_delta.T, axis=0)
    
    # Delta2
    mfccs_delta2 = librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=src, n_mfcc=13), order=2)
    mfccs_delta2 = np.mean(mfccs_delta2.T, axis=0)
    
    # Prosodic Energy (RMS)
    rms = librosa.feature.rms(y=audio)
    mean_rms = np.mean(rms, axis=1)
    
    # Gabungkan semua fitur
    features = np.concatenate([mfccs, mfccs_delta, mfccs_delta2, mean_chroma, mean_rms])
    return features

def extract_features2(file_path):
    audio, src = librosa.load(file_path)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=src, n_mfcc=20)
    mfccs = np.mean(mfccs.T, axis=0)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=src)
    mean_chroma = np.mean(chroma, axis=1)
    
    # Delta
    mfccs_delta = librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=src, n_mfcc=20))
    mfccs_delta = np.mean(mfccs_delta.T, axis=0)
    
    # Delta2
    mfccs_delta2 = librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=src, n_mfcc=20), order=2)
    mfccs_delta2 = np.mean(mfccs_delta2.T, axis=0)
    
    # Prosodic Energy (RMS)
    rms = librosa.feature.rms(y=audio)
    mean_rms = np.mean(rms, axis=1)
    
    # Gabungkan semua fitur
    features = np.concatenate([mfccs, mfccs_delta, mfccs_delta2, mean_chroma, mean_rms])
    return features

# Fungsi untuk ekstraksi fitur dari file suara
def extract_features3(file_path):
    audio, src = librosa.load(file_path)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=src, n_mfcc=30)
    mfccs = np.mean(mfccs.T, axis=0)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=src)
    mean_chroma = np.mean(chroma, axis=1)
    
    # Delta
    mfccs_delta = librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=src, n_mfcc=30))
    mfccs_delta = np.mean(mfccs_delta.T, axis=0)
    
    # Delta2
    mfccs_delta2 = librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=src, n_mfcc=30), order=2)
    mfccs_delta2 = np.mean(mfccs_delta2.T, axis=0)
    
    # Prosodic Energy (RMS)
    rms = librosa.feature.rms(y=audio)
    mean_rms = np.mean(rms, axis=1)
    
    # Gabungkan semua fitur
    features = np.concatenate([mfccs, mfccs_delta, mfccs_delta2, mean_chroma, mean_rms])
    return features

def normalize_audio(audio):
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude == 0:
        return audio
    normalized_audio = audio / max_amplitude
    return (normalized_audio * 32767).astype(np.int16)

def add_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def remove_info():
  time.sleep(5)
  st.empty()

def home_page():
    add_background("static/images/bluebg.jpg")
    st.title("EmoVoice")
    st.subheader("EmoVoice adalah sistem pengenalan emosi berbasis suara (Speech Emotion Recognition")
    st.markdown("<p style='font-size: 24px;'>Sistem ini mampu mendeteksi tujuh emosi dasar:</p>", unsafe_allow_html=True)
    st.write("<p style='font-size: 24px; color: maroon'>• Takut (Fear)<br>• Terkejut (Surprise)<br>• Sedih (Sad)<br>• Marah (Angry)<br>• Jijik (Disgust)<br>• Bahagia (Happy)<br>• Netral (Neutral)</p>", unsafe_allow_html=True)

def emotion_detection():
    st.title("Emotion Detection")
    st.subheader("Kenali Emosi Berdasarkan Suaramu")
    add_background("static/images/bluebg.jpg")

    # Opsi model yang tersedia
    model_option = st.selectbox(
        "Pilih Model Prediksi",
        options=["KNN", "Random_Forest", "Decision_Tree"]  # Perbaiki konsistensi nama
    )

    # Opsi jumlah fitur yang akan diekstraksi
    feature_option = st.selectbox(
        "Pilih Jumlah Fitur Ekstraksi",
        options=[13, 20, 30]
    )

    # Pilihan fungsi ekstraksi fitur
    if feature_option == 13:
        extract_features = extract_features1
    elif feature_option == 20:
        extract_features = extract_features2
    elif feature_option == 30:
        extract_features = extract_features3

    if st.session_state.audio_file is not None:
        st.audio(st.session_state.audio_file, format="audio/wav")
        st.success("Audio Berhasil Direkam")

        if st.button("Muat Ulang"):
            st.session_state.audio_file = None
            st.rerun()

    if st.session_state.audio_file is None:
        sound_file = st.file_uploader("Unggah File Suara (WAV, MP3)", type=["wav", "mp3"])
        if sound_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(sound_file.read())
            st.session_state.audio_file = temp_file.name
            st.success("File audio berhasil diunggah.")

    if st.session_state.audio_file and st.button("Submit Audio"):
        try:
            # Ekstraksi fitur dari file audio
            features = extract_features(st.session_state.audio_file)

            # Load model dan scaler jika diperlukan
            model_path = f'model_{model_option.lower()}_{feature_option}.joblib'

            if model_option == "KNN":
                scaler_path = f'scaler_{model_option.lower()}_{feature_option}.joblib'
                scaler = joblib.load(scaler_path)

            model = joblib.load(model_path)

            # Data dari model
            if model_option == "KNN":
                train_data = model["features"]
                train_labels = model["labels"]

                # Standarisasi fitur
                features_scaled = scaler.transform([features])

                # Prediksi emosi menggunakan fungsi knn_predict
                predicted_emotion = knn_predict(train_data, train_labels, features_scaled[0], k=1)

            elif model_option == "Random_Forest":
                # Tidak ada standarisasi untuk Random Forest
                predicted_emotion = model.prediksi(features)
                # Peta label emosi
                label_emosi = {
                0: 'angry',
                1: 'disgust',
                2: 'fear',
                3: 'happy',
                4: 'neutral',
                5: 'sad',
                6: 'surprise'
                }

                # Konversi prediksi ke label emosi
                predicted_emotion = label_emosi[predicted_emotion]


            elif model_option == "Decision_Tree":
                # Pastikan features berbentuk 2D
                features = np.array([features])

                # Prediksi menggunakan Decision Tree
                predicted_emotion = model.predict(features)[0]  # Ambil prediksi pertama
                label_emosi = {
                    0: 'angry',
                    1: 'disgust',
                    2: 'fear',
                    3: 'happy',
                    4: 'neutral',
                    5: 'sad',
                    6: 'surprise'
                }

                # Konversi prediksi ke label emosi
                predicted_emotion = label_emosi[int(predicted_emotion)]  # Konversi ke integer jika diperlukan


            else:
                st.error("Model tidak dikenali. Pilih model yang valid.")
                return

            st.success(f"Emosi yang terdeteksi: {predicted_emotion}")
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam memproses audio: {str(e)}")


def article_page():
    st.title("Ruang Baca")
    add_background("static/images/bluebg.jpg")

    # List of articles
    articles = {
        "Memahami Pengenalan Emosi dalam Suara": "article_1_page",
        "Pentingnya Deteksi Emosi dalam Interaksi Manusia-Mesin": "article_2_page",
        "Kemajuan Teknologi Pengenalan Emosi dalam Suara": "article_3_page",
        "Aplikasi Pengenalan Emosi dalam Kesehatan Mental": "article_4_page",
        "Tantangan dan Masa Depan Deteksi Emosi": "article_5_page"
    }

    # Create buttons for each article
    for title, page in articles.items():
        if st.button(title):
            st.session_state.current_page = page
            st.rerun()  # Redirect to the new page

def article_1_page():
    st.title("Artikel 1: Memahami Pengenalan Emosi dalam Suara")
    add_background("static/images/bluebg.jpg")
    st.write("""
        Pengenalan emosi dalam suara adalah salah satu cabang menarik dari pengolahan sinyal audio dan kecerdasan buatan. Sistem ini berupaya untuk mengenali emosi manusia 
        melalui pola suara yang diucapkan. Pengenalan emosi sering melibatkan analisis komponen akustik seperti nada, kecepatan bicara, dan intonasi.

        Dalam konteks teknologi, pengenalan emosi tidak hanya terbatas pada aplikasi hiburan, tetapi juga memiliki dampak signifikan pada sektor seperti layanan pelanggan, 
        kesehatan mental, dan pendidikan. Algoritma seperti K-Nearest Neighbors (KNN), Support Vector Machines (SVM), serta metode deep learning seperti Convolutional Neural 
        Networks (CNN) telah banyak digunakan untuk mengembangkan sistem ini. 

        Selain itu, fitur suara seperti Mel-frequency Cepstral Coefficients (MFCC), Chroma, dan Delta memberikan data penting untuk memahami karakteristik akustik 
        yang mendukung deteksi emosi. EmoVoice mengintegrasikan pendekatan ini untuk memberikan hasil analisis yang akurat dan real-time.
    """)
    st.write("""
        Implementasi pengenalan emosi dalam suara menghadapi tantangan teknis, seperti pengaruh kebisingan latar belakang, perbedaan bahasa, dan variasi emosi 
        antar-individu. Namun, dengan pendekatan berbasis data dan peningkatan algoritma, teknologi ini terus berkembang untuk memenuhi kebutuhan pengguna modern.

        Salah satu pencapaian utama dalam proyek seperti EmoVoice adalah memberikan pengalaman yang intuitif dan menyenangkan bagi pengguna, 
        memungkinkan mereka untuk lebih memahami emosi melalui suara mereka sendiri.
    """)
    if st.button("Kembali ke Daftar Artikel"):
        st.session_state.current_page = "first"
        st.rerun()

def article_2_page():
    st.title("Artikel 2: Pentingnya Deteksi Emosi dalam Interaksi Manusia-Mesin")
    add_background("static/images/bluebg.jpg")
    st.write("""
        Deteksi emosi dalam interaksi manusia-mesin adalah bidang yang terus berkembang. Dengan memahami emosi pengguna, sistem AI dapat memberikan 
        tanggapan yang lebih personal dan relevan. Hal ini meningkatkan pengalaman pengguna secara keseluruhan.

        Dalam aplikasi layanan pelanggan, misalnya, AI yang mampu mengenali emosi seperti frustrasi atau kebahagiaan dapat membantu menangani permintaan 
        pelanggan dengan lebih efektif. Sebagai contoh, jika sistem mendeteksi kemarahan dalam suara pelanggan, maka AI dapat merespons dengan nada yang lebih 
        empatik dan menyelesaikan masalah dengan cara yang lebih diplomatis.

        Di sektor pendidikan, deteksi emosi dapat membantu pengajar memahami kesulitan siswa, terutama dalam pembelajaran jarak jauh. Teknologi ini 
        memungkinkan sistem untuk memberikan dukungan tambahan jika emosi seperti kebingungan atau kelelahan terdeteksi. 
    """)
    st.write("""
        Proyek EmoVoice dirancang untuk menghadirkan kemampuan seperti ini, membuka peluang baru bagi teknologi untuk meningkatkan hubungan 
        antara manusia dan mesin. Teknologi yang memahami emosi manusia tidak hanya memperkaya pengalaman pengguna, tetapi juga berkontribusi 
        pada inovasi yang lebih luas di berbagai sektor.
    """)
    if st.button("Kembali ke Daftar Artikel"):
        st.session_state.current_page = "first"
        st.rerun()

def article_3_page():
    st.title("Artikel 3: Kemajuan Teknologi Pengenalan Emosi dalam Suara")
    add_background("static/images/bluebg.jpg")
    st.write("""
        Teknologi pengenalan emosi suara telah mengalami banyak kemajuan dalam beberapa tahun terakhir. Kemunculan algoritma deep learning 
        seperti Recurrent Neural Networks (RNN) dan Convolutional Neural Networks (CNN) telah membawa peningkatan signifikan pada akurasi dan 
        keandalan sistem ini.

        Salah satu inovasi penting adalah integrasi data berbasis cloud, memungkinkan analisis suara secara real-time dari berbagai perangkat. 
        Dengan pendekatan ini, sistem dapat menangani volume data besar tanpa mengorbankan kinerja. Hal ini sangat relevan untuk aplikasi yang 
        membutuhkan waktu respons cepat, seperti chatbot atau layanan pelanggan.

        Selain itu, teknologi pengenalan emosi semakin fokus pada generalisasi lintas bahasa dan budaya. Penelitian menunjukkan bahwa emosi 
        dapat diekspresikan secara berbeda di setiap budaya, sehingga algoritma harus dirancang untuk memahami nuansa ini. Dalam proyek EmoVoice, 
        kami memastikan bahwa teknologi kami dapat diadaptasi untuk berbagai konteks global.
    """)
    st.write("""
        Masa depan pengenalan emosi dalam suara sangat menjanjikan, dengan potensi untuk diterapkan dalam bidang kesehatan mental, hiburan, 
        dan bahkan keamanan. Dengan terus berinovasi, EmoVoice bertujuan untuk menjadi pelopor dalam teknologi ini.
    """)
    if st.button("Kembali ke Daftar Artikel"):
        st.session_state.current_page = "first"
        st.rerun()

def article_4_page():
    st.title("Artikel 4: Aplikasi Pengenalan Emosi dalam Kesehatan Mental")
    add_background("static/images/bluebg.jpg")
    st.write("""
        Kesehatan mental adalah salah satu bidang yang dapat sangat diuntungkan oleh teknologi pengenalan emosi. Dengan menganalisis 
        pola suara seseorang, teknologi ini dapat memberikan wawasan tentang kondisi emosional individu, membantu dalam diagnosis 
        dan pemantauan kondisi seperti depresi, kecemasan, atau stres.

        Sistem seperti EmoVoice dapat digunakan oleh terapis untuk melacak perubahan emosi pasien dari waktu ke waktu, memberikan data 
        objektif yang mendukung perawatan. Selain itu, dalam situasi darurat, sistem dapat mendeteksi tanda-tanda bahaya, seperti 
        nada suara yang mencerminkan keputusasaan, dan memberi peringatan kepada pihak berwenang.
    """)
    st.write("""
        Namun, penerapan teknologi ini juga harus memperhatikan aspek privasi dan etika. EmoVoice berkomitmen untuk memastikan bahwa 
        data pengguna aman dan hanya digunakan untuk tujuan yang telah disetujui.

        Dengan pengembangan lebih lanjut, pengenalan emosi dalam kesehatan mental dapat menjadi alat yang sangat kuat untuk mendukung 
        masyarakat yang lebih sehat dan lebih sadar akan pentingnya kesehatan emosional.
    """)
    if st.button("Kembali ke Daftar Artikel"):
        st.session_state.current_page = "first"
        st.rerun()

def article_5_page():
    st.title("Artikel 5: Tantangan dan Masa Depan Deteksi Emosi")
    add_background("static/images/bluebg.jpg")
    st.write("""
        Deteksi emosi menghadapi sejumlah tantangan yang perlu diatasi untuk mencapai potensinya yang maksimal. Salah satu tantangan 
        utama adalah keragaman emosi antar-individu dan budaya. Sistem harus mampu mengenali perbedaan ini untuk memberikan hasil 
        yang akurat di berbagai konteks.

        Tantangan lainnya adalah kebisingan latar belakang dan kualitas suara yang buruk, yang dapat memengaruhi akurasi analisis. 
        Solusi potensial melibatkan penggunaan filter audio canggih dan algoritma pembelajaran mendalam untuk mengurangi dampak faktor 
        eksternal ini.
    """)
    st.write("""
        Masa depan deteksi emosi sangat menjanjikan dengan kemungkinan integrasi ke dalam perangkat wearable, seperti jam tangan pintar 
        atau earbud, untuk pemantauan emosi secara real-time. Hal ini dapat membuka peluang baru dalam bidang seperti olahraga, 
        kesehatan, dan hiburan.

        EmoVoice berkomitmen untuk mengatasi tantangan ini dan terus berinovasi untuk memastikan bahwa teknologi deteksi emosi dapat 
        digunakan secara luas dan bermanfaat bagi masyarakat global.
    """)
    if st.button("Kembali ke Daftar Artikel"):
        st.session_state.current_page = "first"
        st.rerun()



def about_page():
    st.title("About Us")
    add_background("static/images/bluebg.jpg")
    
    st.subheader("People Behind The EmoVoice Project")
    st.write(
        """The creator of EmoVoice is an undergraduate 3rd semester data science student who is passionate about harnessing the power of Artificial Intelligence. 
        With a keen interest in machine learning and audio processing, they embarked on this project to explore the fascinating intersection of technology and human emotion. 
        EmoVoice aims to revolutionize the way we understand emotions through voice, providing insights that can enhance communication and empathy in various applications. 
        By leveraging advanced algorithms, particularly the K-Nearest Neighbors (KNN) method, and state-of-the-art audio analysis techniques, this project seeks to accurately identify and classify a range of human emotions. 
        The feature extraction process includes Mel-frequency cepstral coefficients (MFCC), Chroma features, and their delta variations (MFCC Delta 1 and Delta 2), paving the way for more intuitive human-computer interactions."""
    )
    st.write(
        '''Driven by curiosity and a desire to innovate, this project is a stepping stone towards a future where machines can better understand and respond to human feelings. 
        The implications of such technology are vast, from improving customer service experiences to aiding mental health professionals in understanding their clients better. 
        Join us on this exciting journey as we delve into the world of emotion recognition and its potential to transform interactions in our daily lives. 
        With EmoVoice, we aspire to create a tool that not only recognizes emotions but also fosters deeper connections between people and technology, 
        ultimately contributing to a more empathetic and understanding society.'''
    )

    st.markdown("---")
    st.subheader("Our Team")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("static/images/rusa.jpg", caption="Muhammad Rifat Syarief ( 23031554 ) rifat@mhs.unesa.ac.id")

    with col2:
        st.image("static/images/faiz.jpg", caption="Moch Faiz Febriawan ( 23031554068 )  mochfaiz.23068@mhs.unesa.ac.id")

    with col3:
        st.image("static/images/serigala.jpg", caption="Alamsyah Ramadhan Vaganza ( 23031554 ) amalsyah@mhs.unesa.ac.id")

    st.markdown("---")
    st.subheader("Emovoice 2024")
    st.markdown("---")

pages = {
    "Main": {
        "Home": home_page,
        "Emotion Detection": emotion_detection
    },
    "Resource": {
        "Literasi": article_page,
        "About Us": about_page
    }
}

def main():
    st.set_page_config(page_title="Audio Recorder", page_icon="🎙️", layout="wide")

    if 'page' not in st.session_state:
        st.session_state.page = 'first'
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = None

    if st.session_state.page == 'first':
        add_background("static/images/bluebg.jpg")
        st.title("Selamat Datang Di EmoVoice")
        st.write("Kenali Emosi Berdasarkan Suaramu")
        if st.button("Mulai Program"):
            st.session_state.page = 'home'
    else:
        if st.session_state.current_page == "article_page":
            article_page()
        elif st.session_state.current_page == "article_1_page":
            article_1_page()
        elif st.session_state.current_page == "article_2_page":
            article_2_page()
        elif st.session_state.current_page == "article_3_page":
            article_3_page()
        elif st.session_state.current_page == "article_4_page":
            article_4_page()
        elif st.session_state.current_page == "article_5_page":
            article_5_page()
        else :
            selected = st.sidebar.selectbox("Navigation", list(pages.keys()), key='nav')
            sub_selected = st.sidebar.selectbox("Page", list(pages[selected].keys()), key='subnav')

            current_page = f"{selected}_{sub_selected}"
            if st.session_state.current_page != current_page:
                st.session_state.current_page = current_page
                st.session_state.audio_file = None
                st.rerun()

            pages[selected][sub_selected]()
    
if __name__ == "__main__":
    main()
