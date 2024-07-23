from itertools import combinations

names = ['Esha', 'Anjali', 'Bhalchandra', 'Shibangi', 'Asha']

def getNamesCombination():
    la, lb, lc = [], [], []
    
    for idx in range(1, len(names) +1):
        la.append(list(combinations(names, idx)))
        
    lb = [list(e) for le in la for e in le] 
    
    lc = [','.join(v) for v in lb] 
    
    return lc  