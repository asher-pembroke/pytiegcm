from tiegcm.tiegcm import TIEGCM, Model_Manager
import pandas as pd

test_directory = '~/Downloads/2013.03.01.tie-gcm.data/'

mm = Model_Manager(test_directory)

print mm.file_times.values()


xlat = -3.69857
xlon = -156.48846
xalt = 360.10342*1e5 #cm

mm.file_times.values()

times = pd.date_range(start ='2013-03-01 00:20:00', 
                      end = '2013-03-01 08:00:00', 
                      freq = '1S')

result = [mm.density(xlat, xlon, xalt,  t) for t in times]

