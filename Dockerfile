# Gunakan base image Python dengan versi yang kompatibel
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Buat direktori kerja di dalam container
WORKDIR /app

# install lib
RUN pip install tensorflow keras numpy flask pillow 

# Salin semua file backend ke dalam container
COPY . .

# Expose port Flask
EXPOSE 5000

# Perintah untuk menjalankan aplikasi
CMD ["python", "asd.py"]
