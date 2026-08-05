import smtplib
import subprocess
from email.mime.image import MIMEImage
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
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Attach text body
    msg.attach(MIMEText(html_content, "html"))

    # 3. Load image from memory (simulated via byte array / open binary)
    # Replace this dummy byte read with your actual in-memory image generation/bytes
    fake_image_bytes = b""  
    # Or if reading an existing memory stream/variable: 
    with open(img_path, "rb") as f: fake_image_bytes = f.read()

    image_part = MIMEImage(fake_image_bytes, name="attachment.png")
    image_part.add_header("Content-Disposition", "attachment", filename="attachment.png")
    msg.attach(image_part)

    # 4. Pipe raw email string directly into Himalaya CLI
    raw_email = msg.as_string()
    process = subprocess.run(
        ["himalaya", "template", "send"],
        input=raw_email,
        text=True,
        capture_output=True,
        check=True
    )

    print("Email sent via Himalaya:", process.stdout)