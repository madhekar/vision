
import piexif


def setImgMetadata(img_path, s_date_time_original, s_user_comment):
  result = "no changed"
  try:
    exif_dict = piexif.load(img_path)

    if s_date_time_original:
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = s_date_time_original.encode("ascii")
        result = "success"


    if s_user_comment:
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = s_user_comment.encode("ascii")
        result = "success"
    
    if result == "success":
        piexif.insert(piexif.dump(exif_dict), img_path)  

  except Exception as e:
     print(f"Exception creatring datetime original and user comment in  {img_path}")  
     result = "failed"

  return result  

if __name__=="__main__":
    ip = "/home/madhekar/temp/travel/switzerland/images/alain-rouiller-YDsG9gJC3N4-unsplash.jpg"
    sdt ="2025:09:12 12:12:12"
    suc = "people"

    setImgMetadata(ip, sdt, suc)

    print(f"---> {piexif.load(ip)}")