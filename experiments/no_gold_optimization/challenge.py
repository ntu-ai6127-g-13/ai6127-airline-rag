"""Synthetic diagnostics from the project taxonomy, never natural gold or fitting rows.

Calibration and monitor use different sentence templates. These artificial labels
cannot establish real-world accuracy. No model is fitted on these sentences.
"""
import pandas as pd

TOPICS={
 'seat':('seat','comfortable','cramped'),
 'staff':('cabin attendant','courteous','rude'),
 'food':('onboard meal','delicious','inedible'),
 'entertainment':('movie selection','excellent','disappointing'),
 'ground':('boarding process','efficient','chaotic'),
 'value':('ticket price','reasonable','excessive'),
 'overall':('overall experience','wonderful','terrible'),
 'other':('aircraft toilet','spotless','filthy'),
}
FACTS={
 'seat':('The seats in row {n} have a pitch of thirty-two inches.','My allocated seat number is {n}C.'),
 'staff':('There were {n} flight attendants working in the cabin.','The cabin crew consisted of {n} people.'),
 'food':('The meal tray contained {n} bread rolls.','Lunch service began at {n}:15.'),
 'entertainment':('The entertainment menu listed {n} documentaries.','There were {n} films on the seatback entertainment screen.'),
 'ground':('Boarding began at gate {n}.','My checked baggage weighed {n} kilograms.'),
 'value':('The return fare came to {n} hundred dollars.','I paid {n} hundred euros for the ticket.'),
 'other':('The aircraft carried registration number XY-{n}.','Our itinerary contained {n} flight segments.'),
}


def generate(excluded_norm,normalize):
    rows=[]
    def add(text,aspect,label,role,family):
        if normalize(text) in excluded_norm:return
        rows.append({'chunk_id':f'synthetic_{len(rows):04d}','review_id':f'{role}_{family}',
                     'text':text,'aspect':aspect,'subjectivity':'objective' if label=='objective' else 'subjective',
                     'polarity':None if label=='objective' else label,'label':label,
                     'role':role,'source':'synthetic_taxonomy_diagnostic_not_gold','family':family})
    for aspect,(noun,pos,neg) in TOPICS.items():
        for label,adj in [('positive',pos),('negative',neg)]:
            for j,text in enumerate([f'I found the {noun} {adj} on this journey.',
                                      f'In my opinion, the {noun} was really {adj}.',
                                      f'The {noun} struck me as {adj}.',
                                      f'For this trip, I would describe the {noun} as {adj}.']):
                add(text,aspect,label,'calibrate' if j<2 else 'monitor',f'evaluation_{j}')
        # Negation tests use clear reversal of favorable attributes, not "not bad" litotes.
        for j,text in enumerate([f'The {noun} was not {pos} at all.',f'I cannot describe the {noun} as {pos}.']):
            add(text,aspect,'negative','calibrate' if j==0 else 'monitor',f'negation_{j}')
    for aspect,templates in FACTS.items():
        for role,idx,values in [('calibrate',0,[3,5,7,9]),('monitor',1,[4,6,8,12])]:
            for n in values:add(templates[idx].format(n=n),aspect,'objective',role,f'fact_{aspect}_{idx}')
    # Aspect-specific contrasts follow the documented last-contrast-clause rule.
    pairs=[('seat','food'),('staff','ground'),('food','value'),('entertainment','seat'),('other','staff'),('value','entertainment')]
    for a,b in pairs:
        na,pa,nea=TOPICS[a];nb,pb,neb=TOPICS[b]
        add(f'The {na} was {pa}, but the {nb} was {neb}.',b,'negative','calibrate','contrast_but')
        add(f'The {na} was {nea}; however, the {nb} was {pb}.',b,'positive','monitor','contrast_however')
    f=pd.DataFrame(rows)
    assert f.text.map(normalize).is_unique
    assert not set(f.loc[f.role.eq('calibrate'),'review_id'])&set(f.loc[f.role.eq('monitor'),'review_id'])
    return f
