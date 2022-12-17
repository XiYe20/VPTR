DATALINK="https://drive.google.com/uc?export=download&id=12FqH2KyH4CSlmddhfpRBoAYcyEB92CbE"
wget -P ./ DATALINK

mkdir rbc_data
python3 ./data_gen.py