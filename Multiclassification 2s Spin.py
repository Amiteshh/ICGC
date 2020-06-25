"""a=['5','10','15','20','5','10','15','20','50','100','150','200','250','300','350','400','450','500','.5','.8','.5','.5','.8','.5','60','80','120','50','UIE','CIE','250','260','270','280','Glitch','I','II','III','10','20','30','30','60','Noise']
print(a[43])"""

next_val=0
#------------------------------------------------------------------
#                    TRAINING DATASETS PREPRATION               ---
#------------------------------------------------------------------

class dataprep_train:

#------------------------------------------------------------------
#                    SIMULATED SIGNALS+TRANSIENT NOISE
#------------------------------------------------------------------
  def simulated_signals(noise):  
    global next_val,signal_gw
    apx=['TaylorT1', 'TaylorT2', 'SpinTaylorT4','SpinTaylorT5','EOBNRv2' , 'SEOBNRv1', 'SEOBNRv2','IMRPhenomB']
    with open('gdrive/My Drive/GW data/labels_mixed.csv', 'a', newline='') as file:
      for a in tqdm(range(len(apx))):
        check=np.zeros(noise.shape[1])
        k=0
        spin1x,spin1y,spin1z,spin2x,spin2y,spin2z=0,0,0,0,0,0
        if apx=='TaylorT1':
          for m1 in range(5,21,5):
            for m2 in range(5,21,5):
              for d in range(50,501,50):
                for fl in [40,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)

        elif apx=='TaylorT2':
          for m1 in range(5,16,5):
            for m2 in range(5,16,5):
              for d in range(50,501,50):
                for fl in [40,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)
                    
        elif apx=='SpinTaylorT4' or apx=='SpinTaylorT5':
          for m1 in range(5,16,5):
            for m2 in range(5,16,5):
              for d in range(50,501,50):
                for fl in [60,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      spin1x=0.5,spin1y=0.8,spin1z=0.5,
                                      spin2x=0.5,spin2y=0.8,spin2z=0.5,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)
          
        elif apx=='EOBNRv2' or apx=='SEOBNRv1' or apx=='SEOBNRv2':
          for m1 in range(5,16,5):
            for m2 in range(5,16,5):
              for d in range(50,501,50):
                for fl in [40,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)
        else:
          for m1 in range(5,16,5):
            for m2 in range(5,16,5):
              for d in range(50,501,50):
                for fl in [40,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)
                    if len(hp)<=16384:
                      signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise[:,::2])
                      pos=np.random.randint(0,signal_gw.shape[1]-int(np.ceil(len(hp)/2)))
                      signal_gw[next_val:next_val+noise.shape[0],pos:pos+int(np.ceil(len(hp)/2))]+=hp[::2]
                      writer = csv.writer(file)
                      for i in range(noise.shape[0]):
                        col=np.zeros(44)
                        if apx=='SpinTaylorT4' or apx=='SpinTaylorT5':
                          col[m1//5],col[3+m2//4],col[7+d//50],col[18],col[19],col[20],col[21],col[22],col[23],col[23+fl//40],col[27]=1,1,1,1,1,1,1,1,1,1,1
                        else:
                          col[m1//5],col[3+m2//4],col[7+d//50],col[23+fl//40],col[27]=1,1,1,1,1
                        col[43]=1
                        writer.writerow(col)
                      k=k+1
                      next_val+=(noise.shape[0])
                    
#------------------------------------------------------------------
#                    ECHOES +TRANSIENT NOISE
#------------------------------------------------------------------
  def echoes(noise):
    global next_val,signal_gw
    signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise[:,::2])
    with open('gdrive/My Drive/GW data/labels_mixed.csv', 'a', newline='') as file:
      for loop in tqdm(range(10)):##############################10
        t=np.linspace(0,.3,np.random.randint(16384))
        y1,y2=np.zeros(len(t)),np.zeros(len(t))
        i=0
        for j in range(8):
          for f in [250,260,270,280]:
            for i in range(len(t)):
              aa=t[i]-0.0295-j*0.0295
              y1[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*f*aa)
            pos=np.random.randint(0,signal_gw.shape[1]-np.floor(len(t)/2))
            signal_gw[next_val:next_val+noise.shape[0]-1,pos:pos+int(np.ceil(len(t)/2))]+=y1[::2]
            next_val+=(noise.shape[0])
            writer = csv.writer(file)
            for i in range(noise.shape[0]):
              col=np.zeros(44)
              col[28],col[29+(f//10)-24],col[43]=1,1,1
              writer.writerow(col)

        r=.3
        for j in range(8):
          for f in [250,260,270,280]:
            for i in range(len(t)):
              aa=t[i]-0.0295-j*0.0295-(j*(j+1)/2)*r*0.0295
              y2[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*f*aa)
            pos=np.random.randint(0,signal_gw.shape[1]-np.floor(len(t)/2))
            signal_gw[next_val:next_val+noise.shape[0]-1,pos:pos+int(np.ceil(len(t)/2))]+=y2[::2]
            next_val+=(noise.shape[0])
            writer = csv.writer(file)          
            for i in range(noise.shape[0]):
                  col=np.zeros(44)
                  col[29],col[29+(f//10)-24],col[43]=1,1,1
                  writer.writerow(col)

#------------------------------------------------------------------
#                    GLITCHES+TRANSIENT NOISE
#------------------------------------------------------------------
  def glitches(noise):
    global next_val,signal_gw
    signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise[:,::2])
    with open('gdrive/My Drive/GW data/labels_mixed.csv', 'a', newline='') as file:
      for i in tqdm(['helix2','whistle','wandering_line','violin_mode','Tomte','koi_fish','scratchy','scattered_light','repeating_blip','power_line','paired_dove','low_freq_burst','light_modulation']):
        loc='gdrive/My Drive/GW data/Glitches/'+i+'.wav'
        rate,data=s.read(loc)
        for c1 in range(5,9):
          for c2 in range(3,13,2):
            b, a = signal.butter(c1, .9, btype='lowpass', analog=False)
            low_passed = signal.filtfilt(b, a, data)
            y = 10e-28*signal.medfilt(low_passed,c2)
            k=np.zeros(8820)
            j=0
            for i in range(0,len(y),5):
              k[j]=y[i]
              j+=1
            pos=np.random.randint(0,signal_gw.shape[1]-int(np.floor(len(k)/2)))
            signal_gw[next_val:next_val+noise.shape[0],pos:pos+int(np.floor(len(k)/2))]+=k[::2]
            next_val+=(noise.shape[0])
            writer = csv.writer(file)
            for i in range(noise.shape[0]):
              col=np.zeros(44)
              col[34],col[43]=1,1
              writer.writerow(col)

#------------------------------------------------------------------------------
#                             CCSNe+Transient Noise               -------------
#------------------------------------------------------------------------------
  def ccsne(noise):
    global next_val,signal_gw
    signal_gw[next_val:next_val+noise.shape[0]]=np.copy(noise[:,::2])
    with open('gdrive/My Drive/GW data/labels_mixed.csv', 'a', newline='') as file:
      val=['signal_A1B1G1_R.dat','signal_A1B2G1_R.dat','signal_A1B3G1_R.dat','signal_A1B3G2_R.dat','signal_A1B3G3_R.dat','signal_A1B3G5_R.dat','signal_A2B4G1_R.dat','signal_A3B1G1_R.dat','signal_A3B2G1_R.dat','signal_A3B2G2_R.dat','signal_A3B2G4_soft_R.dat','signal_A3B2G4_R.dat','signal_A3B3G1_R.dat','signal_A3B3G2_R.dat','signal_A3B3G3_R.dat','signal_A3B3G5_R.dat','signal_A3B4G2_R.dat','signal_A3B5G4_R.dat','signal_A4B1G1_R.dat','signal_A4B1G2_R.dat','signal_A4B2G2_R.dat','signal_A4B2G3_R.dat','signal_A4B4G4_R.dat','signal_A4B4G5_R.dat','signal_A4B5G4_R.dat','signal_A4B5G5_R.dat']
      for aak in tqdm(val):
        loc='gdrive/My Drive/GW data/CCSNe/'+aak
        x, y = np.loadtxt(loc,unpack=True, usecols=[0,1])
        for r in [10,20,30]:
          for theta in [30,60]:
            y = 1/8*np.sqrt(15/np.pi)*y/r*(np.sin(theta))**2
            new_arr=np.zeros(16000)
            j=0
            for i in range(0,len(y),5):
              new_arr[j]=y[i]
              j+=1
            pos=np.random.randint(0,signal_gw.shape[1]-np.ceil(len(new_arr)/2))
            signal_gw[next_val:next_val+noise.shape[0],pos:pos+int(np.ceil(len(new_arr)/2))]+=new_arr[::2]
            writer = csv.writer(file)
            for i in range(noise.shape[0]):
              col=np.zeros(44)
              if val=='signal_A2B4G1_R.dat'  or   val=='signal_A3B3G5_R.dat' or  val=='signal_A3B5G4_R.dat' :
                col[37]=1
              elif val=='signal_A3B3G1_R.dat'  or val=='signal_A3B4G2_R.dat':
                col[36]=1
              elif val=='signal_A3B3G2_R.dat' or  val=='signal_A3B3G3_R.dat' or  val=='signal_A4B2G2_R.dat' or  val=='signal_A4B2G3_R.dat' or  val=='signal_A4B4G4_R.dat' or  val=='signal_A4B4G5_R.dat' or  val=='signal_A4B5G4_R.dat' or  val=='signal_A4B5G5_R.dat':
                col[36],col[35]=1,1
              else:
                col[35]=1
              col[37+r//10],col[40+theta//30],col[43]=1,1,1
              writer.writerow(col)
            next_val=next_val+noise.shape[0]    

#------------------------------------------------------------------------------
#                             MIXED SIGNALS                       -------------
#------------------------------------------------------------------------------
  def mixed_signals_BHBNSB(noise):
    global next_val,signal_gw
    apx=['TaylorT1', 'TaylorT2', 'SpinTaylorT4', 'SpinTaylorT5','EOBNRv2' , 'SEOBNRv1', 'SEOBNRv2','IMRPhenomB']
    with open('gdrive/My Drive/GW data/labels_mixed.csv', 'a', newline='') as file:
      for a in tqdm(range(len(apx))):
        check=np.zeros(noise.shape[1])
        k=0
        spin1x,spin1y,spin1z,spin2x,spin2y,spin2z=0,0,0,0,0,0
        if apx=='TaylorT1':
          for m1 in range(5,21,5):
            for m2 in range(5,21,5):
              for d in range(50,501,50):
                for fl in [40,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)

        elif apx=='TaylorT2':
          for m1 in range(5,16,5):
            for m2 in range(5,16,5):
              for d in range(50,501,50):
                for fl in [40,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)
                    
        elif apx=='SpinTaylorT4' or apx=='SpinTaylorT5':
          for m1 in range(5,16,5):
            for m2 in range(5,16,5):
              for d in range(50,501,50):
                for fl in [60,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      spin1x=0.5,spin1y=0.8,spin1z=0.5,
                                      spin2x=0.5,spin2y=0.8,spin2z=0.5,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)
          
        elif apx=='EOBNRv2' or apx=='SEOBNRv1' or apx=='SEOBNRv2':
          for m1 in range(5,16,5):
            for m2 in range(5,16,5):
              for d in range(50,501,50):
                for fl in [40,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)
        else:
          for m1 in range(5,16,5):
            for m2 in range(5,16,5):
              for d in range(50,501,50):
                for fl in [40,80,120]:
                  if (m1+m2+d+fl) not in check:
                    check[k]=m1+m2+d+fl
                    hp,hc = get_td_waveform(approximant=apx[a],
                                      mass1=m1,mass2=m2,
                                      delta_t=1.0/4096,
                                      f_lower=fl,f_final=50, 
                                      distance=d)
                    if len(hp)<=16385:
                      for loops in range(2):####10
                        t=np.linspace(0,.3,np.random.randint(16384))
                        y2=np.zeros(len(t))          
                        r=.3
                        for j in range(3,8):
                          for i in range(len(t)):
                            aa=t[i]-0.0295-j*0.0295-(j*(j+1)/2)*r*0.0295
                            y2[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*250*aa)

                          #wave=['helix2']#,'whistle','wandering_line','violin_mode','Tomte']:#,'koi_fish','scratchy','scattered_light','repeating_blip','power_line','paired_dove','low_freq_burst','light_modulation']:
                          #for wave_name in wave: 
                          #  loc='gdrive/My Drive/GW data/Glitches/'+wave_name+'.wav'
                          #  rate,data=s.read(loc)
                          #  for c2 in [5,7]:
                          #    b, a = signal.butter(7, .9, btype='lowpass', analog=False)
                          #    low_passed = signal.filtfilt(b, a, data)
                          #    z = 10e-28*signal.medfilt(low_passed,c2)
                          #    sig=np.zeros(8820)
                          #    jj=0
                          #    for ii in range(0,len(z),5):
                          #      sig[jj]=z[ii]
                          #      jj+=1

                          signal_gw[next_val:next_val+noise.shape[0]]=np.copy(noise[:,::2])
                          pos=np.random.randint(0,signal_gw.shape[1]-int(np.ceil(len(hp)/2)))
                          signal_gw[next_val:next_val+noise.shape[0] ,pos:pos+int(np.ceil(len(hp)/2))]+=hp[::2]
                          #pos=np.random.randint(0,signal_gw.shape[1]-np.floor(len(sig))
                          #signal_gw[next_val:next_val+noise.shape[0] ,pos:pos+len(sig)]+=sig
                          pos=np.random.randint(0,signal_gw.shape[1]-int(np.floor(len(y2)/2)))
                          signal_gw[next_val:next_val+noise.shape[0] ,pos:pos+int(np.ceil(len(y2)/2))]+=y2[::2]
                          writer = csv.writer(file)
                          for i in range(noise.shape[0]):
                            col=np.zeros(44)
                            if apx=='SpinTaylorT4' or apx=='SpinTaylorT5':
                              col[m1//5],col[3+m2//4],col[7+d//50],col[18],col[19],col[20],col[21],col[22],col[23],col[23+fl//40],col[27]=1,1,1,1,1,1,1,1,1,1,1
                            else:
                              col[m1//5],col[3+m2//4],col[7+d//50],col[23+fl//40],col[27]=1,1,1,1,1
                            col[29],col[29+(25//10)-24]=1,1
                            writer.writerow(col)  
                            k=k+1
                          next_val+=(noise.shape[0])
                                #for i in range(noise.shape[0]):
                                #  col=np.zeros(44)
                                #  if apx=='SpinTaylorT4' or apx=='SpinTaylorT5':
                                #    col[m1//5],col[3+m2//4],col[7+d//50],col[18],col[19],col[20],col[21],col[22],col[23],col[23+fl//40],col[27]=1,1,1,1,1,1,1,1,1,1,1
                                #  else:
                                #    col[m1//5],col[3+m2//4],col[7+d//50],col[23+fl//40],col[27]=1,1,1,1,1
                                #  col[29],col[29+(25//10)-24],col[34]=1,1,1
                                #writer.writerow(col)                         




#------------------------------------------------------------------------------
#                             MIXED SIGNALS                       -------------
#------------------------------------------------------------------------------
  def mixed_signals_CCSNe(noise):
    global next_val,signal_gw  
    with open('gdrive/My Drive/GW data/labels_mixed.csv', 'a', newline='') as file:
      val=['signal_A1B1G1_R.dat','signal_A1B2G1_R.dat','signal_A1B3G1_R.dat','signal_A1B3G2_R.dat','signal_A1B3G3_R.dat','signal_A1B3G5_R.dat','signal_A2B4G1_R.dat','signal_A3B1G1_R.dat','signal_A3B2G1_R.dat','signal_A3B2G2_R.dat','signal_A3B2G4_soft_R.dat','signal_A3B2G4_R.dat','signal_A3B3G1_R.dat','signal_A3B3G2_R.dat','signal_A3B3G3_R.dat','signal_A3B3G5_R.dat','signal_A3B4G2_R.dat','signal_A3B5G4_R.dat','signal_A4B1G1_R.dat','signal_A4B1G2_R.dat','signal_A4B2G2_R.dat','signal_A4B2G3_R.dat','signal_A4B4G4_R.dat','signal_A4B4G5_R.dat','signal_A4B5G4_R.dat','signal_A4B5G5_R.dat']
      for aak in val:
        loc='gdrive/My Drive/GW data/CCSNe/'+aak
        x, y = np.loadtxt(loc,unpack=True, usecols=[0,1])
        for r in [10,20,30]:
          for theta in [30,60]:
            y = 1/8*np.sqrt(15/np.pi)*y/r*(np.sin(theta))**2
            new_arr=np.zeros(16000)
            j=0
            for i in range(0,len(y),5):
              new_arr[j]=y[i]
              j+=1
            wave=['helix2','whistle','wandering_line','violin_mode','Tomte','koi_fish','scratchy','scattered_light','repeating_blip','power_line','paired_dove','low_freq_burst','light_modulation']
            for wave_name in wave: 
              loc='gdrive/My Drive/GW data/Glitches/'+wave_name+'.wav'
              rate,data=s.read(loc)
              for c2 in [5,7]:
                b, a = signal.butter(7, .9, btype='lowpass', analog=False)
                low_passed = signal.filtfilt(b, a, data)
                z = 10e-28*signal.medfilt(low_passed,c2)
                sig=np.zeros(8820)
                jj=0
                for ii in range(0,len(z),5):
                  sig[jj]=z[ii]
                  jj+=1
                signal_gw[next_val:next_val+noise.shape[0]]=np.copy(noise[:,::2])
                pos=np.random.randint(0,signal_gw.shape[1]-int(np.ceil(len(sig)/2)))
                signal_gw[next_val:next_val+noise.shape[0],pos:pos+int(np.ceil(len(sig)/2))]+=sig[::2]
                pos=np.random.randint(0,signal_gw.shape[1]-int(np.floor(len(new_arr)/2)))
                signal_gw[next_val:next_val+noise.shape[0],pos:pos+int(np.floor(len(new_arr)/2))]+=new_arr[::2]
                writer = csv.writer(file)
                for i in range(noise.shape[0]):
                  col=np.zeros(44)
                  if val=='signal_A2B4G1_R.dat' or  val=='signal_A3B3G5_R.dat' or  val=='signal_A3B5G4_R.dat' :
                    col[37]=1
                  elif val=='signal_A3B3G1_R.dat' or val=='signal_A3B4G2_R.dat':
                    col[36]=1
                  elif val=='signal_A3B3G2_R.dat' or  val=='signal_A3B3G3_R.dat' or  val=='signal_A4B2G2_R.dat' or  val=='signal_A4B2G3_R.dat' or  val=='signal_A4B4G4_R.dat' or  val=='signal_A4B4G5_R.dat' or  val=='signal_A4B5G4_R.dat' or  val=='signal_A4B5G5_R.dat':
                    col[36],col[35]=1,1
                  else:
                    col[35]=1
                  col[34],col[37+r//10],col[40+theta//30],col[43]=1,1,1,1
                  writer.writerow(col)
                next_val+=noise.shape[0]



#----------------------------------------------------------------------------------
#                               PIPELINES                                       ---
#----------------------------------------------------------------------------------
def train_pipeline(noise):
  global next_val,signal_gw
  val=dataprep_train
  with open('gdrive/My Drive/GW data/labels_mixed.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['5','10','15','20','5','10','15','20','50','100','150','200','250','300','350','400','450','500','.5','.8','.5','.5','.8','.5','60','80','120','50','UIE','CIE','250','260','270','280','Glitch','I','II','III','10','20','30','30','60','Noise'])
  print('\nSimulated GW .......')    
  val.simulated_signals(noise)
  print('\nSimulated GW training set 100%')
  print('data size :'+ str(next_val)+'\n')  
  print('\nEchoes...')
  val.echoes(noise)
  print('data size :'+ str(next_val)+'\n')
  print('\nEchoes 100%')
  print('\nCCSNE...')  
  val.ccsne(noise)
  print('data size :'+ str(next_val)+'\n')
  print('\nCCSNE 100%')
  print('\nGlitches...')    
  val.glitches(noise)
  print('data size :'+ str(next_val)+'\n')  
  print('\nGlitches 100%')  
  print('\nMixed training BHBNSB ...' )
  val.mixed_signals_BHBNSB(noise)
  print('\nMixed set BHBNSB 100%')
  print('data size :'+ str(next_val)+'\n')
  print('\nMixed training CCSNe...' )
  val.mixed_signals_CCSNe(noise)
  print('\nMixed set CCSNe 100%')
  print('data size :'+ str(next_val)+'\n')  
  hf = h5py.File('gdrive/My Drive/GW data/mixed.h5', 'w')
  hf.create_dataset('mixed', data=signal_gw)
  hf.close()"""
#--------------------------------------------------------
#                           MAIN                    -----
#--------------------------------------------------------
if __name__ == '__main__':
  global next_val
  print('\nCollecting noise.....\n data size : '+str(next_val))
  hf= h5py.File('gdrive/My Drive/GW data/noise.h5', 'r')
  group_key = list(hf.keys())[0]
  noise= hf[group_key]
  noise=np.array(noise)
  print(noise.shape,type(noise))
  hf.close()
  noise=noise[:10]
  print(noise.shape)

  print('\n\nPreparing data..... ')
  train_pipeline(noise) #--------training-------------#
  print('\nPreparing data........100%\n\n')
