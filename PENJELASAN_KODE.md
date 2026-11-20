# Penjelasan Kode Lengkap — `Tugas_NLP_Text_Preprocessing.ipynb`

Dokumen ini menjelaskan alur kerja, fungsi, parameter, dan logika dari **setiap sel kode** pada berkas [`Tugas_NLP_Text_Preprocessing.ipynb`](./Tugas_NLP_Text_Preprocessing.ipynb) secara berurutan dari awal hingga akhir.

- **Nama:** Annada Khairatun Nisa
- **NIM:** 24146038
- **Mata Kuliah:** SIF301 – Natural Language Processing
- **Dataset:** [`dipawidia/ecommerce-product-reviews-sentiment`](https://huggingface.co/datasets/dipawidia/ecommerce-product-reviews-sentiment)

---

## Daftar Isi

1. [Sel Kode 1 — Instalasi Pustaka (`%pip install`)](#sel-kode-1--instalasi-pustaka)
2. [Sel Kode 2 — Import Library & Konfigurasi Awal](#sel-kode-2--import-library--konfigurasi-awal)
3. [Sel Kode 3 — Load Dataset dari Hugging Face](#sel-kode-3--load-dataset-dari-hugging-face)
4. [Sel Kode 4 — Eksplorasi Informasi Umum Dataset](#sel-kode-4--eksplorasi-informasi-umum-dataset)
5. [Sel Kode 5 — Distribusi & Visualisasi Label Sentimen](#sel-kode-5--distribusi--visualisasi-label-sentimen)
6. [Sel Kode 6 — Inisialisasi & Definisi 5 Fungsi Preprocessing](#sel-kode-6--inisialisasi--definisi-5-fungsi-preprocessing)
7. [Sel Kode 7 — Eksekusi Pipeline Preprocessing pada Seluruh Data](#sel-kode-7--eksekusi-pipeline-preprocessing-pada-seluruh-data)
8. [Sel Kode 8 — Demonstrasi Perubahan Bertahap (Indeks 0)](#sel-kode-8--demonstrasi-perubahan-bertahap-indeks-0)
9. [Sel Kode 9 — Menampilkan 10 Contoh Sebelum & Sesudah Preprocessing](#sel-kode-9--menampilkan-10-contoh-sebelum--sesudah-preprocessing)
10. [Sel Kode 10 — Perhitungan Ringkasan Kuantitatif Dampak Preprocessing](#sel-kode-10--perhitungan-ringkasan-kuantitatif-dampak-preprocessing)

---

## Sel Kode 1 — Instalasi Pustaka

```python
%pip install -q datasets pandas matplotlib nltk Sastrawi
```

### Penjelasan
- `%pip install` adalah perintah *magic command* di Jupyter/Colab untuk memasang pustaka langsung ke kernel Python yang sedang aktif.
- Opsi `-q` (*quiet*) menyembunyikan log instalasi panjang agar tampilan output tetap bersih.
- Pustaka yang dipasang:
  - `datasets`: mengambil dataset dari Hugging Face Hub.
  - `pandas`: pengolahan tabel (`DataFrame`).
  - `matplotlib`: pembuatan grafik batang dan diagram lingkaran.
  - `nltk`: tokenisasi (`word_tokenize`) dan kamus *stopword* bahasa Indonesia.
  - `Sastrawi`: *stopword* dan *stemmer* khusus morfologi bahasa Indonesia.

---

## Sel Kode 2 — Import Library & Konfigurasi Awal

```python
import re
import time
from functools import lru_cache

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from IPython.display import display
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Unduh resource NLTK yang diperlukan secara diam-diam
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Atur tampilan tabel pandas agar teks panjang tidak terpotong
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

print("Semua library berhasil diimpor!")
```

### Penjelasan
1. **Modul Standar Python:**
   - `re`: dipakai pada tahap *cleaning* untuk menghapus pola URL, angka, tanda baca, dan simbol dengan *regular expression*.
   - `time`: dipakai untuk mengukur lama waktu eksekusi *preprocessing*.
   - `lru_cache` (dari `functools`): menyimpan hasil *stemming* kata yang sudah pernah diproses agar tidak dihitung berulang kali.
2. **Unduhan Resource NLTK:**
   - `punkt` dan `punkt_tab` dibutuhkan oleh `word_tokenize` (keduanya diunduh agar kompatibel dengan NLTK versi lama maupun versi terbaru ≥ 3.8.2).
   - `stopwords` berisi daftar kata umum bahasa Indonesia.
   - Dibungkus `try/except` agar program tetap berjalan tanpa error bila salah satu resource sudah tersedia atau versi NLTK berbeda.
3. **Pengaturan Tampilan `pandas`:**
   - `display.max_colwidth = None` memastikan isi ulasan panjang tidak terpotong menjadi tanda `...` saat ditampilkan dalam tabel.
   - `display.max_columns = None` memastikan seluruh kolom tampil lengkap.

---

## Sel Kode 3 — Load Dataset dari Hugging Face

```python
# Memuat dataset dari Hugging Face
dataset = load_dataset("dipawidia/ecommerce-product-reviews-sentiment")
print("Struktur Dataset Hugging Face:")
print(dataset)

# Menggabungkan seluruh split (train/test/dst.) menjadi satu DataFrame
dfs = []
print("\nRincian jumlah data per split:")
for split_name, split_data in dataset.items():
    temp_df = split_data.to_pandas()
    dfs.append(temp_df)
    print(f"- Split '{split_name}': {len(temp_df):,} baris")

df = pd.concat(dfs, ignore_index=True)

# Identifikasi nama kolom teks dan label sentimen
TEXT_COL = "review" if "review" in df.columns else df.columns[0]
LABEL_COL = "sentimen" if "sentimen" in df.columns else df.columns[-1]

print(f"\nTotal seluruh data: {len(df):,} baris")
print(f"Kolom teks yang digunakan : '{TEXT_COL}'")
print(f"Kolom label sentimen      : '{LABEL_COL}'")
```

### Penjelasan
- `load_dataset(...)` mengunduh berkas `labeledReview.datasetFix.json` dari Hugging Face dan mengembalikannya sebagai objek `DatasetDict`.
- Perulangan `for split_name, split_data in dataset.items()` mengubah setiap *split* menjadi `pandas.DataFrame` lalu menggabungkannya dengan `pd.concat(..., ignore_index=True)` agar penomoran indeks baris rapi dari `0` sampai `11.605`.
- Penetapan `TEXT_COL` dan `LABEL_COL` secara dinamis mengarahkan kode ke kolom `review` (teks ulasan Indonesia) dan `sentimen` (label `0`/`1`) sambil menyediakan *fallback* jika struktur kolom berubah.

**Output Aktual:**
- 1 *split* (`train`) berisi **11.606 baris** dengan 3 kolom: `['review', 'translate', 'sentimen']`.

---

## Sel Kode 4 — Eksplorasi Informasi Umum Dataset

```python
print("=" * 60)
print("INFORMASI UMUM DATASET")
print("=" * 60)
print(f"Jumlah Baris (Data) : {df.shape[0]:,}")
print(f"Jumlah Kolom        : {df.shape[1]}")
print(f"Nama Kolom          : {list(df.columns)}")
print("\nTipe Data & Nilai Non-Null:")
df.info()

print("\nJumlah Missing Values per Kolom:")
print(df.isnull().sum())

print(f"\nJumlah Data Duplikat (berdasarkan kolom '{TEXT_COL}'): {df.duplicated(subset=[TEXT_COL]).sum():,}")

print("\n5 Contoh Data Pertama:")
display(df.head())
```

### Penjelasan
- `df.shape`: mengambil jumlah baris (`df.shape[0]`) dan kolom (`df.shape[1]`).
- `df.info()`: menampilkan nama kolom, jumlah nilai *non-null*, tipe data (`object` untuk teks, `int64` untuk label), serta pemakaian memori (`272.1+ KB`).
- `df.isnull().sum()`: memeriksa apakah terdapat nilai kosong (*missing values*) di tiap kolom.
- `df.duplicated(subset=[TEXT_COL]).sum()`: menghitung berapa banyak teks ulasan yang muncul ganda.
- `display(df.head())`: menampilkan 5 baris pertama dataset dalam bentuk tabel rapi.

**Output Aktual:**
- **11.606 baris**, **3 kolom**, **0 *missing values*** pada seluruh kolom, dan **7 data duplikat** pada kolom `review`.

---

## Sel Kode 5 — Distribusi & Visualisasi Label Sentimen

```python
# Distribusi Label Sentimen
label_map = {
    0: "Negatif (0)",
    1: "Positif (1)",
    "0": "Negatif (0)",
    "1": "Positif (1)",
}

distribusi = df[LABEL_COL].value_counts().sort_index()
persentase = (df[LABEL_COL].value_counts(normalize=True).sort_index() * 100).round(2)

df_distribusi = pd.DataFrame({
    "Kode Label": distribusi.index,
    "Kategori Sentimen": [label_map.get(idx, str(idx)) for idx in distribusi.index],
    "Jumlah Data": distribusi.values,
    "Persentase (%)": persentase.values,
})

print("Distribusi Label Sentimen:")
display(df_distribusi)
```

### Penjelasan
1. **Perhitungan Distribusi:**
   - `value_counts().sort_index()` menghitung jumlah data untuk tiap label (`0` dan `1`) secara terurut.
   - `value_counts(normalize=True) * 100` mengubah frekuensi absolut menjadi persentase dan dibulatkan 2 desimal (`.round(2)`).
   - `label_map` memetakan kode angka ke teks deskriptif (`Negatif (0)` dan `Positif (1)`), mendukung tipe integer maupun string.
2. **Visualisasi (`matplotlib`):**
   - `plt.subplots(1, 2, figsize=(12, 4.5))` membuat kanvas berisi 2 grafik berdampingan.
   - `axes[0].bar(...)` menggambar **Bar Chart** disertai label teks jumlah dan persentase di atas tiap batang (`axes[0].text(...)`) serta `set_ylim(...)` agar label tidak terpotong di batas atas grafik.
   - `axes[1].pie(...)` menggambar **Pie Chart** dengan `autopct="%1.2f%%"` dan sedikit celah pemisah (`explode=[0.03, 0.03]`).

**Output Aktual:**
- `Negatif (0)`: **5.803** ulasan (**50,00%**)
- `Positif (1)`: **5.803** ulasan (**50,00%**)

---

## Sel Kode 6 — Inisialisasi & Definisi 5 Fungsi Preprocessing

```python
# 1. Inisialisasi daftar stopword Bahasa Indonesia (Sastrawi + NLTK)
sastrawi_stopwords = set(StopWordRemoverFactory().get_stop_words())
try:
    nltk_stopwords = set(stopwords.words("indonesian"))
except LookupError:
    nltk_stopwords = set()

STOPWORDS_ID = sastrawi_stopwords.union(nltk_stopwords)
print(f"Jumlah kamus stopword Bahasa Indonesia: {len(STOPWORDS_ID)} kata")

# 2. Inisialisasi Stemmer Bahasa Indonesia (Sastrawi) dengan caching
stemmer = StemmerFactory().create_stemmer()

@lru_cache(maxsize=None)
def stem_kata(kata: str) -> str:
    return stemmer.stem(kata)
```

### Penjelasan Inisialisasi
- **Penggabungan Stopwords (`union`):** Menggunakan struktur data `set` membuat pencarian kata (`kata not in STOPWORDS_ID`) berjalan dalam waktu konstan $O(1)$, jauh lebih cepat dibandingkan `list`. Gabungan kamus Sastrawi dan NLTK menghasilkan **774 kata stopword unik**.
- **Optimasi Stemming (`@lru_cache(maxsize=None)`):** Algoritma Sastrawi memeriksa kamus dan aturan imbuhan bahasa Indonesia di Python murni. Dengan *caching*, kata yang muncul ribuan kali (misalnya `"barang"`, `"pengiriman"`, `"bagus"`) cukup di-stem satu kali saja, lalu pemanggilan berikutnya langsung mengambil hasil dari memori.

### Penjelasan Masing-Masing Fungsi

#### 1. `case_folding(text)`
```python
def case_folding(text: str) -> str:
    if pd.isna(text):
        return ""
    return str(text).lower()
```
- Mengecek nilai kosong (`pd.isna`) agar tidak memicu `AttributeError`, lalu mengubah seluruh huruf menjadi huruf kecil (`.lower()`).

#### 2. `cleaning(text)`
```python
def cleaning(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```
Setiap baris `re.sub` mengganti pola derau dengan spasi (agar dua kata yang terpisah tanda baca seperti `baret/lecet` tidak menempel menjadi `baretlecet`):

| Pola Regex | Target yang Dihapus | Contoh |
| :--- | :--- | :--- |
| `https?://\S+\|www\.\S+` | Tautan URL (`http`, `https`, `www`) | `https://shopee.co.id/...` → ` ` |
| `<.*?>` | Tag HTML | `<br>`, `<b>` → ` ` |
| `[@#]\w+` | Mention (`@user`) & hashtag (`#tag`) | `@toko`, `#murah` → ` ` |
| `\d+` | Seluruh angka | `13`, `2`, `4` → ` ` |
| `[^a-z\s]` | Tanda baca, emoji, simbol selain huruf `a-z` dan spasi | `😄`, `👍`, `?`, `/`, `..` → ` ` |
| `\s+` + `.strip()` | Spasi ganda/tab/baris baru berlebih di tengah & tepi | `"  bagus   sekali "` → `"bagus sekali"` |

#### 3. `tokenisasi(text)`
```python
def tokenisasi(text: str) -> list:
    if not text:
        return []
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()
```
- Memecah string bersih menjadi daftar token kata menggunakan `nltk.tokenize.word_tokenize`.
- Menyediakan *fallback* aman ke `text.split()` jika data tokenizer NLTK gagal diakses.

#### 4. `stopword_removal(tokens)`
```python
def stopword_removal(tokens: list) -> list:
    return [kata for kata in tokens if kata not in STOPWORDS_ID and len(kata) > 1]
```
- Mempertahankan hanya token yang **tidak ada** dalam `STOPWORDS_ID` dan memiliki panjang lebih dari 1 huruf (`len(kata) > 1`) untuk membuang huruf tunggal sisa penghapusan angka/tanda baca (seperti `"n"` atau `"w"` dari `"W4"`).

#### 5. `stemming(tokens)`
```python
def stemming(tokens: list) -> list:
    return [stem_kata(kata) for kata in tokens]
```
- Mengonversi setiap token ke bentuk kata dasarnya melalui fungsi `stem_kata` yang sudah di-cache.

---

## Sel Kode 7 — Eksekusi Pipeline Preprocessing pada Seluruh Data

```python
start_time = time.time()

df["case_folding"] = df[TEXT_COL].apply(case_folding)
df["cleaned"] = df["case_folding"].apply(cleaning)
df["tokens"] = df["cleaned"].apply(tokenisasi)
df["no_stopwords"] = df["tokens"].apply(stopword_removal)
df["stemmed_tokens"] = df["no_stopwords"].apply(stemming)

# Gabungkan kembali token hasil akhir menjadi string teks bersih
df["teks_preprocessing"] = df["stemmed_tokens"].apply(lambda tokens: " ".join(tokens))

elapsed = time.time() - start_time
print(f"\nPreprocessing selesai untuk {len(df):,} ulasan dalam {elapsed:.2f} detik!")
print(f"Total kata unik yang di-stem (cache): {stem_kata.cache_info().currsize:,} kata")
```

### Penjelasan
- `.apply(...)` menjalankan masing-masing fungsi pada seluruh 11.606 baris dan menyimpan keluaran tiap tahap ke kolom terpisah agar proses mudah diaudit.
- `" ".join(tokens)` menyatukan kembali daftar token hasil *stemming* menjadi satu string utuh di kolom `teks_preprocessing`.
- `stem_kata.cache_info().currsize` melaporkan jumlah kata unik yang benar-benar diproses oleh Sastrawi.

**Output Aktual:**
- Total ulasan diproses: **11.606 ulasan**
- Waktu eksekusi: **664,81 detik**
- Kata unik di-stem (cache): **12.046 kata**

---

## Sel Kode 8 — Demonstrasi Perubahan Bertahap (Indeks 0)

```python
contoh_idx = 0
print(f"0. Teks Asli        : {df.loc[contoh_idx, TEXT_COL]}")
print(f"1. Case Folding     : {df.loc[contoh_idx, 'case_folding']}")
print(f"2. Cleaning         : {df.loc[contoh_idx, 'cleaned']}")
print(f"3. Tokenisasi       : {df.loc[contoh_idx, 'tokens']}")
print(f"4. Stopword Removal : {df.loc[contoh_idx, 'no_stopwords']}")
print(f"5. Stemming         : {df.loc[contoh_idx, 'stemmed_tokens']}")
print(f"   Hasil Akhir Teks : {df.loc[contoh_idx, 'teks_preprocessing']}")
```

### Penjelasan
- Mengambil satu baris (`df.loc[0, ...]`) lalu mencetak hasil dari setiap kolom perantara secara berurutan.
- Tujuannya untuk memverifikasi secara visual bahwa tiap tahap bekerja sesuai fungsi masing-masing (misalnya `"Suara"` → `"suara"`, tanda `.` terhapus, `"dan"/"hanya"/"saya"/"sudah"` dibuang, dan `"pastikan"` berubah menjadi `"pasti"`).

---

## Sel Kode 9 — Menampilkan 10 Contoh Sebelum & Sesudah Preprocessing

```python
sample_parts = []
for label_val in sorted(df[LABEL_COL].dropna().unique()):
    subset = df[df[LABEL_COL] == label_val]
    n_take = min(5, len(subset))
    sample_parts.append(subset.sample(n=n_take, random_state=42))

df_sample = pd.concat(sample_parts, ignore_index=True).head(10)

if len(df_sample) < 10:
    df_sample = df.sample(n=10, random_state=42).reset_index(drop=True)

tabel_perbandingan = pd.DataFrame({
    "No": range(1, len(df_sample) + 1),
    "Sentimen": df_sample[LABEL_COL].map(lambda x: label_map.get(x, str(x))),
    "Sebelum Preprocessing": df_sample[TEXT_COL],
    "Sesudah Preprocessing": df_sample["teks_preprocessing"],
})

try:
    display(
        tabel_perbandingan.style
        .hide(axis="index")
        .set_properties(**{"text-align": "left", "vertical-align": "top"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
    )
except ImportError:
    display(tabel_perbandingan)
```

### Penjelasan
- **Stratified Sampling:** Melakukan perulangan pada tiap label unik (`0` dan `1`) lalu mengambil 5 baris dari masing-masing kelas (`n=5`) sehingga 10 contoh yang ditampilkan seimbang (5 ulasan negatif dan 5 ulasan positif).
- `random_state=42`: mengunci pemilihan acak agar sampel yang muncul selalu sama setiap kali notebook dijalankan ulang (*reproducible*).
- `tabel_perbandingan.style`: menyembunyikan indeks bawaan pandas (`hide(axis="index")`) dan mengatur perataan teks ke kiri atas agar tabel panjang mudah dibaca.
- Blok `try/except ImportError` mencegah error apabila lingkungan Python lokal tidak memasang pustaka `jinja2` yang diperlukan oleh `DataFrame.style`.

---

## Sel Kode 10 — Perhitungan Ringkasan Kuantitatif Dampak Preprocessing

```python
raw_tokens_series = df[TEXT_COL].fillna("").astype(str).apply(lambda s: s.split())
clean_tokens_series = df["stemmed_tokens"]

total_kata_sebelum = raw_tokens_series.apply(len).sum()
total_kata_sesudah = clean_tokens_series.apply(len).sum()

avg_kata_sebelum = raw_tokens_series.apply(len).mean()
avg_kata_sesudah = clean_tokens_series.apply(len).mean()

vocab_sebelum = len(set(kata for tokens in raw_tokens_series for kata in tokens))
vocab_sesudah = len(set(kata for tokens in clean_tokens_series for kata in tokens))

reduksi_kata_pct = (1 - total_kata_sesudah / total_kata_sebelum) * 100
reduksi_vocab_pct = (1 - vocab_sesudah / vocab_sebelum) * 100
```

### Penjelasan
- Membandingkan korpus sebelum *preprocessing* (`raw_tokens_series`, dipisah berdasarkan spasi) dengan korpus hasil akhir (`stemmed_tokens`) berdasarkan tiga metrik utama:
  1. **Total Token:** jumlah keseluruhan kata pada 11.606 ulasan (`182.871` → `128.111`, turun **29,94%**).
  2. **Rata-rata Kata per Ulasan:** panjang rata-rata tiap ulasan (`15,76` → `11,04` kata, turun **29,94%**).
  3. **Ukuran Kosakata Unik (*Vocabulary Size*):** dihitung menggunakan `set(...)` untuk mendapatkan seluruh kata unik (`26.205` → `10.034` kata, turun **61,71%**).
- Hasil kuantitatif ini menjadi bukti terukur bahwa tahapan *preprocessing* secara efektif menyederhanakan dimensi fitur teks tanpa menghilangkan kata-kata kunci utama ulasan.
