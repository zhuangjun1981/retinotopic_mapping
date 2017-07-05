import smtplib
import os
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email.Utils import COMMASPACE, formatdate
from email import Encoders


class MailMessage(MIMEMultipart):
    def __init__(self, sender, recipients, subject, body, files=[]):
        MIMEMultipart.__init__(self)
        self["From"] = sender
        self["To"] = COMMASPACE.join(recipients)
        self["Date"] = formatdate(localtime=True)
        self["Subject"] = subject
        self.attach(MIMEText(body))
        self.attach_files(files)

    def attach_files(self, files):
        for f in files:
            part = MIMEBase('application', "octet-stream")
            part.set_payload(open(f, "rb").read())
            Encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                'attachment; filename="%s"' % os.path.basename(f))
            self.attach(part)


def send(user, password, recipients, subject, body, files=[],
         smtp_host="aicas-1.corp.alleninstitute.org", smtp_port=25):
    # ensure recipients and files are lists
    if type(recipients) is str:
        recipients = [recipients]
    if type(files) is str:
        files = [files]

    msg = MailMessage(user, recipients, subject, body, files)

    # connect to server
    server = smtplib.SMTP()
    server.connect(smtp_host, smtp_port)
    server.ehlo()
    server.starttls()
    server.ehlo()

    # log into server if password is provided
    if password:
        server.login(user, password)

    # send email
    attempt = server.sendmail(user, recipients, msg.as_string())
    #print "Recipient that could not be reached:", attempt
    complete = server.quit()
    #print "Closing connection:", complete


class AIBSMailer(object):
    def __init__(self, user, password,
                 smtp_host="aicas-1.corp.alleninstitute.org",
                 smtp_port=25):
        self.user = user
        self.pw = password
        self.host = smtp_host
        self.port = smtp_port

    def send_message(self, recipients, subject, body, files=[]):
        send(self.user, self.pw, recipients, subject, body, files,
             self.host, self.port)


if __name__ == '__main__':
    pass
