from pypai import PAI

pai = PAI(username='geo', passwd='geoandPAI,7')

pai.submit(exclude=['.mdb', 'jpg', 'pkl', 'txt'])