# Eksperimen SML - Ayunda Putri

Repository ini berisi eksperimen preprocessing data untuk tugas
kelas Membangun Sistem Machine Learning.

# Struktur Folder


SMSML_AYUNDA-PUTRI/Eksperimen_SML_Ayunda-putri/
├── .github/workflows/preprocessing.yml
├── housing_raw.csv
├── preprocessing/
│   ├── automate_Ayunda-putri.py
│   ├── Eksperimen_Ayunda_putri.ipynb
│   └── housing_preprocessed/
├── requirements.txt

# Eksperimen SML – Ayunda Putri

Repository ini berisi eksperimen preprocessing data untuk tugas kelas  
**Membangun Sistem Machine Learning (Dicoding)**.

Seluruh proses preprocessing dijalankan menggunakan Python dan
telah disiapkan agar dapat dijalankan **secara lokal maupun otomatis
menggunakan GitHub Actions**.

---

## Tujuan Proyek
Tujuan dari eksperimen ini adalah melakukan preprocessing dataset
sebagai tahap awal sebelum proses pemodelan machine learning,
sesuai dengan standar yang ditentukan pada kelas Dicoding.

---

## Struktur Folder Proyek

preprocessing/housing_preprocessed/

## Cara Menjalankan Preprocessing Secara Lokal

Pastikan Python telah terinstall, lalu jalankan perintah berikut:

```bash
pip install -r requirements.txt
python preprocessing/automate_Ayunda-putri.py

# Jika berhasil, file hasil preprocessing akan otomatis tersimpan di folder 
housing_preprocessed.

## Automasi Menggunakan GitHub Actions Repository ini telah dilengkapi dengan workflow GitHub Actions yang terdapat pada file:

.github/workflows/preprocessing.yml

# Workflow ini akan menjalankan proses preprocessing secara otomatis setiap kali terdapat perubahan 
(push) ke branch main.

# Kesimpulan
Dengan adanya preprocessing otomatis ini, proses persiapan data
menjadi lebih konsisten, terstruktur, dan siap digunakan
untuk tahap pemodelan machine learning selanjutnya.

# Author

Ayunda Putri
Eksperimen untuk kelas Membangun Sistem Machine Learning – Dicoding

