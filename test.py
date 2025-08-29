import gdown

# Thay FILE_ID bằng ID thực tế của file ZIP
file_id = "1ZpdVMuEFWDgGlTcN2Z52oMtbbVpG0llT"
url = f"https://drive.google.com/uc?id={file_id}"

output = "hrsid.zip"  # Đặt tên file khi lưu về máy

gdown.download(url, output, quiet=False)
print("Tải file ZIP thành công!")