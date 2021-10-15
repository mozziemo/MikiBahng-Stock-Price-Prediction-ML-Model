from credentials import testmail
import smtplib, ssl

port = 465 # for SSL
smtp_server = testmail.smtp_server
sender_email = testmail.email
receiver_email = testmail.email
password = testmail.password

def sendEmailNotice(email_subject, email_message, port=port, sender_email=sender_email, receiver_email=receiver_email, password=password):
    message = f'Subject: {email_subject} \n\n {email_message}'
    context = ssl.create_default_context()
    
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        try:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
            print('Email sent successfully!')
        except Exception as e:
            print('Email sending failed...........')
            print(repr(e)) # print error message to stdout


if __name__ == '__main__':
    sendEmailNotice('test subject', 'test notice message')

