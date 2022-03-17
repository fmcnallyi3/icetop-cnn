from itertools import product, permutations
import numpy as np
import os

#debug gen data prep
def generate_data_prep(q = None, t = None,  normed = None, reco = None, cosz = None):
    #parameters specific combos
    default_options = { 'q' :  ["mean","sum","product","min","max", None], 
                        't' : ["mean","sum","product","min","max", None, False], #this may be wrong, check in datatool
                        'normed' : [True, False],
                        'reco' : [ None, "plane", "laputop", "small"],
                        'cosz' : [False, True]}

    if q!=None:
        if q=="None":
            default_options["q"]=[None]
        else:
            default_options["q"]=[q]
    if t!=None:
        if t=="None":
            default_options["t"]=[None]
        else:
            default_options["t"]=[t]
    if normed!=None:
        default_options["normed"]=[normed]
    if reco!=None:
        if reco=="None":
            default_options["reco"]=[None]
        else:
            default_options["reco"]=[reco]
    if cosz!=None:
        if cosz==True:
            default_options["cosz"]=[True]
            if None in default_options["reco"]:
                default_options["reco"].remove(None)
        else:
            default_options["cosz"]=[False]

    data_preps = []
#if cosz==True, reco !=None excluse cosz==True and reco==None
    for charge in default_options["q"]:
        for time in default_options["t"]:
            for norm in default_options["normed"]:
                for rec in default_options["reco"]:
                    for cos in default_options["cosz"]:
                        #if ( (rec!=None and cosz!=True) ): #impossible cases go here
                        data_preps.append({"q": charge, "t": time, 
                                                                    "normed": norm, "reco": rec,"cosz": cosz })
                        #else:
                            #print('cant print this compbo')
    
    """
    for charge in default_options["q"]:
        data_preps.append({"q": charge, "t": None, "normed": True, "reco": "plane","cosz":True})
    for time in default_options["t"]:
        data_preps.append({"q": None, "t": time, "normed": True, "reco": "plane","cosz":True})
    for norm in default_options["normed"]:
        data_preps.append({"q": None, "t": None, "normed": norm, "reco": "plane","cosz":True})
    for rec in default_options["reco"]:
        data_preps.append({"q": None, "t": None, "normed": True, "reco": rec,"cosz":True})
    for cos in default_options["cosz"]:
        data_preps.append({"q": None, "t": None, "normed": True, "reco": "plane","cosz":cos})
    """

    #data_preps = product()
    return data_preps

temp = generate_data_prep(t="None", normed=True, reco=None, cosz=False) #cosz=True breaks the code
print(temp)

def compileModel(name, q=None, t=None, normed=False, reco=None, cosz=False):
    
    ##4 layers | t=none & q = none
    #if data_prep["t"] == None & data_prep["q"] == None:
    #    dimensions=4
    ##3 layers | t!=none & t!= false & q = none
    #elif data_prep["t"] == False:
    #    if data_prep["q"] == None:
    #        dimensions=2
    #    else:
    #        dimensions=1
    #else:
    #    dimensions=3
    ##3 layers | t=none & q != none
    ##2 layers | t=false & q = none
    ##1 layer | t=false & q != none

    
    if t is None:
        tdim = 2
    elif not t:
        tdim = 0
    else:
        tdim = 1

    if q is None:
        qdim = 2
    elif q is False:
        qdim = 0
    else:
        qdim = 1


    #actual model layers
    ##CHARGE
    if qdim != 0:
        charge_input = keras.Input(shape=(10,10,qdim,),name="charge")
        #charge_input=keras.Input(shape=(10,10,2,))
        convq1_layer = layers.Conv2D(64,kernel_size=3,padding='same',activation='relu')(charge_input)
        convq2_layer = layers.Conv2D(32,kernel_size=3,padding='same',activation='relu')(convq1_layer)
        convq3_layer = layers.Conv2D(16, kernel_size=3, padding='same',activation="relu")(convq2_layer)
        flatq_layer = layers.Flatten()(convq3_layer)

    ##TIME
    if tdim != 0:
        time_input = keras.Input(shape=(10,10,tdim,),name="time")
        #time_input=keras.Input(shape=(10,10,2,))
        convt1_layer = layers.Conv2D(64,kernel_size=3,padding='same',activation='relu')(time_input)
        convt2_layer = layers.Conv2D(32,kernel_size=3,padding='same',activation='relu')(convt1_layer)
        convt3_layer = layers.Conv2D(16, kernel_size=3, padding='same',activation="relu")(convt2_layer)
        flatt_layer = layers.Flatten()(convt3_layer)

    if not (reco is None):
        zenith_input=keras.Input(shape=(1,))
        if tdim != 0:
            if qdim != 0:
                concat_layer = layers.Concatenate()([flatq_layer,flatt_layer,zenith_input])
                dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
            else:
                concat_layer = layers.Concatenate()([flatt_layer,zenith_input])
                dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
        elif qdim != 0:
            concat_layer = layers.Concatenate()([flatq_layer,zenith_input])
            dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
        else:
            dense1_layer = layers.Dense(256,activation='relu')(zenith_input)
    else:
        if tdim != 0:
            if qdim != 0:
                concat_layer = layers.Concatenate()([flatq_layer,flatt_layer])
                dense1_layer = layers.Dense(256,activation='relu')(concat_layer)
            else:
                dense1_layer = layers.Dense(256,activation='relu')(flatt_layer)
        elif qdim != 0:
            dense1_layer = layers.Dense(256,activation='relu')(flatq_layer)
        else:
            print("All inputs removed!")
            return

    dense2_layer = layers.Dense(256,activation='relu')(dense1_layer)
    dense3_layer = layers.Dense(256,activation="relu")(dense2_layer)
    output = layers.Dense(1)(dense3_layer)

    if not (reco is None):
        if tdim != 0:
            if qdim != 0:
                model = models.Model(inputs=[charge_input,time_input,zenith_input],outputs=output,name=name)
            else:
                model = models.Model(inputs=[time_input,zenith_input],outputs=output,name=name)
        elif qdim != 0:
            model = models.Model(inputs=[charge_input,zenith_input],outputs=output,name=name)
        else:
            model = models.Model(inputs=[zenith_input],outputs=output,name=name)
    else:
        if tdim != 0:
            if qdim != 0:
                model = models.Model(inputs=[charge_input,time_input],outputs=output,name=name)
            else:
                model = models.Model(inputs=[time_input],outputs=output,name=name)
        elif qdim != 0:
            model = models.Model(inputs=[charge_input],outputs=output,name=name)
        else:
            print("All inputs removed! Also, first check didn't trigger!")
            return

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'])
    return model

num=0
while(os.path.isdir('/Home/mays_k/icetop-cnn/Architecture_testers/trainedModels%i' %num)):
    num+=1
print(num)