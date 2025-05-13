from deepface import DeepFace
import pandas as pd
import cv2
"""
[{

'emotion': {'angry': 3.4567246287586784e-08, 'disgust': 1.9612233947950572e-15, 'fear': 1.502852018314205e-13, 'happy': 99.99358654022217, 'sad': 2.2328298932006163e-08, 'surprise': 1.134383602677258e-10, 'neutral': 0.006418131670216098}, 
'dominant_emotion': 'happy', 

'region': {'x': 194, 'y': 192, 'w': 261, 'h': 261, 'left_eye': None, 'right_eye': None}, 
'face_confidence': 0.94, 
'age': 33, 

'gender': {'Woman': 9.052027016878128, 'Man': 90.94797372817993}, 
'dominant_gender': 'Man', 

'race': {'asian': 6.72480876593642, 'indian': 45.579749082311935, 'black': 1.7910822430898097, 'white': 2.767263145283648, 'middle eastern': 3.725002555549297, 'latino hispanic': 39.41208955121574},
 'dominant_race': 'indian'

 }]


"""
#img_path = '/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/Anjali Garba 2018.jpg'
img_path = '/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/IMAG2285.jpg'
#img_path = "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/IMG-20190111-WA0010.jpg"
#img_path = "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/IMAG2478.jpg"
#img_path = "/home/madhekar/work/home-media-app/data/input-data/img/chicago 012.jpg"
def compute_aggregate_msg(in_arr):
    if in_arr:
        if len(in_arr) > 0:
            df = pd.DataFrame(in_arr, columns=['age','emotion','gender','race'])
            print(df.head())
            #age range

            #common emotion 

            #male count vs female count

            #race common race

def detect_human_attributs(img_path):
    people= []
    age, emotion, gender, race = None, None, None, None
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        preds = DeepFace.analyze(img, enforce_detection=True)

        if preds:
            num_faces = len(preds)
            if num_faces > 0:
                for nf in range(num_faces):
                    age = preds[nf]['age']
                    emotion = preds[nf]['dominant_emotion']
                    gender = preds[nf]["dominant_gender"]
                    race = preds[nf]["dominant_race"]
                    #print(f'{img_path}: {nf} of {num_faces} age: {age} - emotion: {emotion} - gender: {gender} - race: {race}')
                    people.append({'age':age, 'emotion': emotion, 'gender': gender, 'race': race})    
    except Exception as e:
        print(f'Error occured in emotion detection: {e}')
    return people

compute_aggregate_msg(detect_human_attributs(img_path))

#print(preds)
#print(f"Age: {preds[0]['age']} Gender: {preds[0]['dominant_gender']}")


"""
[
{'emotion': {'angry': 78.97913330669006, 'disgust': 4.1185593503561784e-08, 'fear': 0.0039451863436185996, 'happy': 1.9555425112420688e-06, 'sad': 7.283648537614854, 'surprise': 1.6235879928446958e-09, 'neutral': 13.733274563150081}, 
'dominant_emotion': 'angry', 
'region': {'x': 838, 'y': 178, 'w': 54, 'h': 54, 'left_eye': None, 'right_eye': None}, 
'face_confidence': 0.96, 
'age': 30, 
'gender': {'Woman': 14.994405210018158, 'Man': 85.00559329986572}, 
'dominant_gender': 'Man', 
'race': {'asian': 0.1823703176341951, 'indian': 0.24059407878667116, 'black': 0.2544892951846123, 'white': 96.30071520805359, 'middle eastern': 0.9613897651433945, 'latino hispanic': 2.0604416728019714}, 
'dominant_race': 'white'
},

{'emotion': {'angry': 39.44425582885742, 'disgust': 1.9847188337251964e-07, 'fear': 0.009656411566538736, 'happy': 0.04504503158386797, 'sad': 3.623424470424652, 'surprise': 4.8008722330905584e-06, 'neutral': 56.87761902809143},
'dominant_emotion': 'neutral', 
'region': {'x': 1002, 'y': 180, 'w': 93, 'h': 93, 'left_eye': None, 'right_eye': None}, 
'face_confidence': 0.93, 
'age': 23, 
'gender': {'Woman': 0.1283219433389604, 'Man': 99.87168312072754}, 
'dominant_gender': 'Man', 
'race': {'asian': 72.66210047458405, 'indian': 0.8193651168805222, 'black': 0.1325396197241998, 'white': 18.952842777628568, 'middle eastern': 1.1745316438935227, 'latino hispanic': 6.258614360259251}, 
'dominant_race': 'asian'
}, 

{'emotion': {'angry': 0.0004344532435928563, 'disgust': 1.5638434514948373e-08, 'fear': 0.00018337245907333003, 'happy': 96.42385287000502, 'sad': 2.5200149765592843, 'surprise': 3.1617119372759525e-06, 'neutral': 1.0555060253434152}, 
'dominant_emotion': 'happy', 
'region': {'x': 1435, 'y': 249, 'w': 98, 'h': 98, 'left_eye': None, 'right_eye': None}, 
'face_confidence': 0.94, 
'age': 26, 
'gender': {'Woman': 0.10235049994662404, 'Man': 99.8976469039917}, 
'dominant_gender': 'Man', 
'race': {'asian': 0.08750495738290759, 'indian': 0.617783396433783, 'black': 0.025304779304180527, 'white': 85.06731568964533, 'middle eastern': 8.643231255622952, 'latino hispanic': 5.558854051368946}, 
'dominant_race': 'white'}, 

{'emotion': {'angry': 0.0006627660636104672, 'disgust': 1.953861234496175e-10, 'fear': 0.0004903503687597565, 'happy': 50.00896751934096, 'sad': 0.10566054308998842, 'surprise': 2.9109303370848745e-05, 'neutral': 49.88419114853091}, 
'dominant_emotion': 'happy', 
'region': {'x': 604, 'y': 336, 'w': 141, 'h': 141, 'left_eye': None, 'right_eye': None}, 
'face_confidence': 0.91, 
'age': 28, 
'gender': {'Woman': 64.85835909843445, 'Man': 35.14164388179779}, 
'dominant_gender': 'Woman', 
'race': {'asian': 33.39528143405914, 'indian': 13.248653709888458, 'black': 2.7383608743548393, 'white': 4.761163517832756, 'middle eastern': 2.508596144616604, 'latino hispanic': 43.34794282913208}, 
'dominant_race': 'latino hispanic'}, 

{'emotion': {'angry': 0.615433556959033, 'disgust': 0.10618981905281544, 'fear': 22.376306354999542, 'happy': 0.004299059946788475, 'sad': 76.72196626663208, 'surprise': 1.0449574716631105e-05, 'neutral': 0.17579442355781794}, 
'dominant_emotion': 'sad', 
'region': {'x': 783, 'y': 309, 'w': 102, 'h': 102, 'left_eye': None, 'right_eye': None}, 
'face_confidence': 0.92, 'age': 34, 
'gender': {'Woman': 6.296107918024063, 'Man': 93.70389580726624}, 
'dominant_gender': 'Man', 
'race': {'asian': 8.749479055404663, 'indian': 19.55879181623459, 'black': 5.4861582815647125, 'white': 12.57595419883728, 'middle eastern': 10.38706749677658, 'latino hispanic': 43.242546916007996}, 
'dominant_race': 'latino hispanic'}, 

{'emotion': {'angry': 0.290802214294672, 'disgust': 0.00019763128875638358, 'fear': 0.0811485864687711, 'happy': 22.034655511379242, 'sad': 5.285521596670151, 'surprise': 0.00013884109648643062, 'neutral': 72.30753302574158}, 
'dominant_emotion': 'neutral', 'region': {'x': 855, 'y': 455, 'w': 168, 'h': 168, 'left_eye': None, 'right_eye': None}, 
'face_confidence': 0.93, 'age': 32, 
'gender': {'Woman': 35.908713936805725, 'Man': 64.09129500389099}, 
'dominant_gender': 'Man', 
'race': {'asian': 29.76360377831504, 'indian': 16.219239380268977, 'black': 5.045917859186871, 'white': 16.545418387105272, 'middle eastern': 9.143074946770065, 'latino hispanic': 23.282751236289887}, 
'dominant_race': 'asian'}, 

{'emotion': {'angry': 0.0020049279555678368, 'disgust': 7.64559246533951e-12, 'fear': 3.3409615074475596e-07, 'happy': 99.9956727027893, 'sad': 7.744985168756102e-05, 'surprise': 1.0701250996447698e-07, 'neutral': 0.0022473557692137547}, 
'dominant_emotion': 'happy', 
'region': {'x': 1123, 'y': 408, 'w': 148, 'h': 148, 'left_eye': None, 'right_eye': None}, 
'face_confidence': 0.96, 'age': 32, 
'gender': {'Woman': 9.3462273478508, 'Man': 90.65377116203308}, 
'dominant_gender': 'Man', 
'race': {'asian': 34.322257895604245, 'indian': 15.81793670761208, 'black': 9.42618083513861, 'white': 7.478207795513141, 'middle eastern': 4.84751715347273, 'latino hispanic': 28.1079011027754}, 
'dominant_race': 'asian'}, 

{'emotion': {'angry': 7.25444282423382e-07, 'disgust': 6.596364622059314e-12, 'fear': 1.958620154951518e-07, 'happy': 99.83112812042236, 'sad': 6.717880474127469e-07, 'surprise': 0.0022033796994946897, 'neutral': 0.16666979063302279},
 'dominant_emotion': 'happy', 
 'region': {'x': 426, 'y': 668, 'w': 350, 'h': 350, 'left_eye': None, 'right_eye': None}, 
 'face_confidence': 0.92, 
 'age': 48, 
 'gender': {'Woman': 0.14843916287645698, 'Man': 99.85156655311584},
 'dominant_gender': 'Man', 
 'race': {'asian': 10.655401647090912, 'indian': 18.795745074748993, 'black': 2.239445596933365, 'white': 18.249088525772095, 'middle eastern': 23.24763387441635, 'latino hispanic': 26.812684535980225},
 'dominant_race': 'latino hispanic'}, 
 
 {'emotion': {'angry': 0.004458322487733999, 'disgust': 1.4800245580629097e-11, 'fear': 99.9163270096389, 'happy': 2.2975072436837585e-07, 'sad': 0.0789794685697444, 'surprise': 9.837239571524542e-09, 'neutral': 0.00022662181463258595}, 'dominant_emotion': 'fear', 'region': {'x': 994, 'y': 1019, 'w': 60, 'h': 60, 'left_eye': None, 'right_eye': None}, 'face_confidence': 0.99, 'age': 30, 'gender': {'Woman': 18.220113217830658, 'Man': 81.77988529205322}, 'dominant_gender': 'Man', 'race': {'asian': 34.81321334838867, 'indian': 0.6515759974718094, 'black': 0.2673488575965166, 'white': 50.61400532722473, 'middle eastern': 3.8001950830221176, 'latino hispanic': 9.85366627573967}, 'dominant_race': 'white'}]

"""