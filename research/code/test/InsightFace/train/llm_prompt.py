import pandas as pd

ld = [
    {
        "age": 34,
        "gender": "Female",
        "cnoun": "woman",
        "name": "unknown",
        "loc": (953, 514),
        "emotion": "happy",
    },
    {
        "age": 47,
        "gender": "Male",
        "cnoun": "man",
        "name": "unknown",
        "loc": (857, 218),
        "emotion": "fear",
    },
    {
        "age": 24,
        "gender": "Female",
        "cnoun": "woman",
        "name": "unknown",
        "loc": (872, 389),
        "emotion": "happy",
    },
    {
        "age": 54,
        "gender": "Male",
        "cnoun": "man",
        "name": "Kumar",
        "loc": (280, 283),
        "emotion": "angry",
    },
    {
        "age": 77,
        "gender": "Male",
        "cnoun": "man",
        "name": "Asha",
        "loc": (242, 497),
        "emotion": "fear",
    },
    {
        "age": 50,
        "gender": "Female",
        "cnoun": "woman",
        "name": "unknown",
        "loc": (548, 535),
        "emotion": "happy",
    },
    {
        "age": 48,
        "gender": "Male",
        "cnoun": "man",
        "name": "unknown",
        "loc": (559, 275),
        "emotion": "sad",
    },
]

df = pd.DataFrame(ld)

print(df.head(20))

def count_people(ld):
    cnt_man = 0
    cnt_woman = 0
    cnt_boy = 0
    cnt_girl = 0
    face_kn = 0 
    agg_person = []
    for d in ld:
        if d["name"] == "unknown":
            if d["cnoun"] == "man":
                cnt_man +=1 
            if d["cnoun"] == "woman":
                cnt_woman += 1
            if d["cnoun"] == "boy":
                cnt_boy += 1
            if d["cnoun"] == "girl":
                cnt_girl += 1
        else:
            agg_person.append({"type": "known", "name":d["name"] ,"cnoun" :d["cnoun"], "emotion": d["emotion"], "loc": d["loc"]})
            face_kn +=1
    agg_person.append({"type": "unknown", "cman": cnt_man, "cwoman": cnt_woman, "cboy":cnt_boy, "cgirl":cnt_girl})

    return agg_person

def create_partial_prompt(agg):
    txt = ""
    for d in agg:
        if d['type'] == 'known':
                s = f"Face at coordinates {d['loc']} is of \"{d['name']}\", a \"{d['cnoun']}\" is expressing \"{d['emotion']}\" emotion. "
                txt += s
        if d['type'] == "unknown":

            if d['cman'] > 0:
                if d['cman'] > 1:
                    s = f"other {d['cman']} men in the image. "
                else: 
                    s = "other one  man in the image. "   
                txt += s    

            if d['cwoman'] > 0:
                if d['cwoman'] > 1:
                    s = f"other {d['cwoman']}  women in the image. "
                else: 
                    s = "other one  woman in the image. "
                txt += s    

            if d["cboy"] > 0:
                if d["cman"] > 1:
                    s = f"other {d['cboy']} boys in the image. "
                else:
                    s = "other one  boy in the image. "
                txt +=s    

            if d['cgirl'] > 0:
                if d['cgirl'] > 1:
                    s = f"other {d['cgirl']}  girls in the image. "
                else: 
                    s = "other one girl in the image. "            
                txt += s
    return txt
aggre = count_people(ld)
print(aggre)
t = create_partial_prompt(aggre)

print(t)