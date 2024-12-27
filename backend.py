import time
import base64
import joblib
import librosa
import tempfile
import numpy as np
import streamlit as st
import sounddevice as sd
from collections import Counter
from scipy.io.wavfile import write, read

SAMPLE_RATE = 44100
DURATION = 3


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Hitung jarak ke semua data latih
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Ambil indeks dari k tetangga terdekat
        k_indices = np.argsort(distances)[:self.k]
        # Ambil label dari k tetangga terdekat
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Voting mayoritas
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def extract_features(file_path):
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

    # Gabungkan semua fitur
    features = np.concatenate([mfccs, mfccs_delta, mfccs_delta2])
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

    audio_path = None
    if st.button("Rekam Suara"):
        st.info(f"Merekam suara selama {DURATION} detik...")
        sd.stop()
        recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        normalized_audio = normalize_audio(recording)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, SAMPLE_RATE, normalized_audio)
        st.session_state.audio_file = temp_file.name

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
            # Extract features
            features = extract_features(st.session_state.audio_file)
            
            # Load model and scaler
            knn = joblib.load('model/knn_model.joblib')
            scaler = joblib.load('model/scaler.joblib')
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Predict emotion
            predicted_emotion = knn.predict(features_scaled)
            
            st.success(f"Emosi yang terdeteksi: {predicted_emotion[0]}")
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


def emobot_page():
    st.title("EmoBot ( Demo )")
    add_background("static/images/bluebg.jpg")

    st.subheader("Chat with EmoBot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Cek jika navigasi dilakukan, hapus riwayat chat
    if "current_page" in st.session_state and st.session_state.current_page != "EmoBot":
        st.session_state.chat_history = []

    user_input = st.text_input("Ketik pesan Anda:", placeholder="Tanyakan sesuatu...")

    if st.button("Kirim"):
        if user_input:
            st.session_state.chat_history.append({"sender": "User", "message": user_input})
            response = emobot_response(user_input)
            st.session_state.chat_history.append({"sender": "EmoBot", "message": response})

    st.markdown("---")
    st.write("### Riwayat Percakapan:")
    for chat in st.session_state.chat_history:
        if chat["sender"] == "User":
            st.markdown(f"**You:** {chat['message']}")
        else:
            st.markdown(f"**EmoBot:** {chat['message']}")

def emobot_response(user_input):
    responses = {
        "hello": "Hii, apakah ada yang bisa dibantu ?",
        "how are you": "I'm just a bot, but I'm here to help!",
        "emotion detection": "I can help detect emotions from your voice! Try it out in the main menu.",
        "assalamu'alaikum" : "Wa'alaikumsalaam.",
        "unesa": "Satu Langkah Di Depan",
        "halo": "Halo! Ada yang bisa saya bantu hari ini?",
        "apa kabar": "Saya hanyalah bot, tapi saya merasa sangat senang bisa membantu Anda!",
        "deteksi emosi": "Saya dapat membantu mendeteksi emosi dari suara Anda! Silakan coba fitur ini di menu utama.",
        "bagaimana cara menggunakan": "Anda dapat menggunakan fitur dengan memilih menu yang tersedia di sidebar.",
        "tentang emovoice": "EmoVoice adalah sistem deteksi emosi berbasis suara. Kami menggunakan teknologi canggih untuk mendeteksi berbagai emosi manusia.",
        "terima kasih": "Sama-sama! Saya senang bisa membantu Anda.",
        "siapa yang membuat emovoice": "EmoVoice dibuat oleh tim mahasiswa yang antusias terhadap kecerdasan buatan dan pengolahan suara.",
        "apa fungsi utama emovoice": "Fungsi utama EmoVoice adalah mendeteksi emosi seperti Bahagia, Sedih, Marah, Takut, Terkejut, Jijik, dan Netral dari suara.",
        "emobot apa kabar": "Saya baik-baik saja! Bagaimana dengan Anda?",
        "bisa bantu saya": "Tentu saja! Silakan tanyakan apa saja.",
        "fitur apa saja di emovoice": "Kami memiliki fitur Deteksi Emosi, Artikel terkait, dan juga interaksi dengan EmoBot seperti ini.",
        "bagaimana cara kerja deteksi emosi": "Kami menggunakan algoritma KNN dan fitur suara seperti MFCC, Chroma, dan Delta untuk mendeteksi emosi dari suara Anda.",
        "kenapa saya harus mencoba ini": "Karena ini adalah cara menarik untuk memahami emosi Anda dan bagaimana teknologi dapat membantu menganalisisnya!",
        "selamat tinggal": "Selamat tinggal! Semoga harimu menyenangkan.",
        "apa itu knn": "KNN atau K-Nearest Neighbors adalah algoritma pembelajaran mesin yang digunakan untuk mengklasifikasikan data berdasarkan tetangga terdekat.",
        "mfcc itu apa": "MFCC atau Mel-frequency cepstral coefficients adalah fitur audio yang sering digunakan untuk analisis suara, seperti deteksi emosi.",
        "chroma itu apa": "Fitur Chroma merepresentasikan energi frekuensi dalam nada musik tertentu dan digunakan dalam pengolahan audio.",
        "bisa ngobrol dengan saya?": "Tentu saja! Saya di sini untuk membantu Anda kapan saja.",
        "siapa yang menggunakan emovoice": "EmoVoice dapat digunakan oleh siapa saja, termasuk profesional kesehatan mental, layanan pelanggan, atau hanya untuk hiburan.",
        "aplikasi ini gratis?": "Ya, Anda bisa menggunakan EmoVoice secara gratis untuk eksplorasi dan penelitian.",
        "apa yang bisa emobot lakukan": "Saya bisa menjawab pertanyaan Anda tentang EmoVoice, memberikan informasi, dan membantu memahami fitur deteksi emosi.",
        "bagaimana cara mendeteksi emosi": "Silakan unggah atau rekam suara Anda di fitur Deteksi Emosi. EmoVoice akan menganalisis suara tersebut untuk mengidentifikasi emosi.",
        "apa kegunaan deteksi emosi": "Deteksi emosi dapat digunakan untuk meningkatkan interaksi manusia-mesin, mendukung kesehatan mental, dan analisis perilaku pengguna.",
        "mengapa emovoice penting": "Karena EmoVoice membantu menjembatani pemahaman antara emosi manusia dan teknologi, membuka peluang baru dalam berbagai aplikasi.",
        "apakah emovoice menggunakan ai": "Ya, EmoVoice menggunakan kecerdasan buatan dan pembelajaran mesin untuk mendeteksi emosi dengan akurat.",
        "di mana emovoice dikembangkan": "EmoVoice dikembangkan oleh mahasiswa yang tertarik dengan pengolahan sinyal suara di Indonesia.",
        "berapa lama deteksi emosi dilakukan": "Proses deteksi emosi biasanya hanya membutuhkan beberapa detik setelah suara diunggah.",
        "apakah hasil deteksi akurat": "Kami menggunakan algoritma yang dioptimalkan untuk akurasi tinggi, namun hasilnya tetap bergantung pada kualitas suara yang diberikan.",
        "apa itu emosi": "Emosi adalah respons psikologis dan fisiologis yang membantu manusia memahami dan berinteraksi dengan dunia sekitar mereka.",
        "bisakah membantu saya memahami algoritma?": "Tentu saja, saya bisa menjelaskan algoritma yang digunakan dalam EmoVoice, seperti KNN dan metode ekstraksi fitur audio.",
        "berapa banyak emosi yang bisa dideteksi?": "EmoVoice mendeteksi 7 emosi dasar: Bahagia, Sedih, Marah, Takut, Jijik, Terkejut, dan Netral."
    }
    return responses.get(user_input.lower(), "Sorry Gw Gapaham, Bisa Di Ulangi ?")


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
        "EmoBot": emobot_page,
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
