import smtplib
from email.encoders import encode_base64
import subprocess
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_with_image(img_path):
    # 1. Define dynamic variables and content
    sender = "bmadhekar@gmail.com"
    recipient = "bmadhekar@gmail.com"
    subject = "ZM update with Image"
    username = "madhekar"

    # Dynamic HTML template in memory
    html_content = f"""
    <html>
    <body>
        <h2>Hello {username},</h2>
        <p>This is a dynamic email generated completely in memory.</p>
        <p>Here is your requested image attachment below.</p>
    </body>
    </html>
    """

    # 2. Build the MIME message structure
    msg = MIMEMultipart('related')
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Attach text body
    msg.attach(MIMEText(html_content, "html"))

    # 3. Load image from memory (simulated via byte array / open binary)
    # Replace this dummy byte read with your actual in-memory image generation/bytes 
    # Or if reading an existing memory stream/variable: 
    with open(img_path, "rb") as f: 
        mime_img = MIMEBase("image", "png") 
        mime_img.set_payload(f.read())

        encode_base64(mime_img)
        #mime_img.add_header('Content-ID', f'{img_path}')
        mime_img.add_header("Content-Disposition", 'attachment', filename=img_path)
        #image_part = MIMEImage(image_bytes, name=img_path)
        #image_part.add_header("Content-Disposition", "inline", filename=img_path)
        msg.attach(mime_img)

        # 4. Pipe raw email string directly into Himalaya CLI
        raw_email = msg.as_string()

        # print(raw_email)
        # s = smtplib.SMTP("localhost")
        # s.send(raw_email)
        process = subprocess.run(
            ["himalaya", "template", "send"],
            input=raw_email,
            text=True,
            capture_output=True,
            check=True
        )

        print("Email sent via Himalaya:", process.stdout)

if __name__=="__main__":
    send_email_with_image("/home/madhekar/tmp/esha/aug_9_esha1.png") # /mnt/zmdata/home-media-app/data/final-data/img/ASSORT_K30/0cff2236-7401-5cc0-b289-7e8db7824acf/IMGP4165.JPG")    