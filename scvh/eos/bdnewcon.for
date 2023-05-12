      PI = ACOS(-1.)
      PI4 = 4*PI
      PI43 = PI4/3
C
C                    --------  PARAMETERS
C
      IMODEL=1
      SHAVE=.0000001
      SHINE=.3
      FRACT=.01
C
      VEOS=1.
      NOBOND=1
      POUT=1.E7
      PBOUND=POUT
      MDOT=0.
      AMASS=0.
      KZONE=0
      TBEGA=0.
      ACMAX=25.
      AMETH=2
      CMETH=1
C
      YHEA=.22
      YDI=5.E-5
      HB=1.
      SPK=12.
      TEFF=3000.
C      MASS IN UNITS OF 10**30 GMS
      MASS=50.
      BMASS=MASS
      NMASS=MASS
C
      FREL=0.
      ITCOR=0
      ILOOP=1
C
      N=120
      NM = N - 1
      NP = N + 1
      N = MIN0 ( MN-1, MAX0 ( N, 30 ) )
C
      OVRFLW=1.E30
      UNDFLW=1./OVRFLW
      LACRET=.FALSE.
      CHECK=.TRUE.
      LSTART=.FALSE.
      STATIC=.FALSE.
      LCORR=.TRUE.
      LCONV=.TRUE.
      LTRANS=.TRUE.
      LPOUT=.TRUE.
      LFILL=.FALSE.
      LCONT=.TRUE.
      LBURN=.FALSE.
      MAG='O'
      GAM='H'
      MAXSTP = 1000
      MINSTP = 0
C
      TBEG=0.000005
      TIMEI = 0.
      TIMEF = 10.
      TIME = TIMEI
      TIMEP = TBEG
      TIMEO = TBEG
      DTIMEO = .02
      DTIMEP = DTIMEO
      STEPP=.2
      STEPO=STEPP
      ISTEP = 0
      JFIRST = 1
C
      ZE = 1.E30
      ZM = 1.E30
      ZT = 1.E10
      ZRAD = 1.E10
      ZP = ZM/(ZRAD*ZT**2)
      ZRAD3 = ZRAD**3
      ZR = ZM/ZRAD3
      ZV=1.
      ACONZ1=1./(4.*PI)*ZM/(ZR*ZRAD**3)
C      ACONZ2=1/(4PI)*ZM**2*G/(ZP*ZRAD**4)
      ACONZ2=5.31036E11
      ACONPS=-(1./3)*LOG(ACONZ1*3.)
C      ACONP1=LN(ZR**(4/3)ZM**(2/3)/ZP*(4PI/3)**(1/3)G/2)
      ACONP1=LOG(5.378572E12)
C      CFLUX=7.56042E-5*ZT*ZRAD**2/ZM
      CFLUX=7.56042E-5
      RCON=8.31434E7
      TCN=1./3.1558E6

C
      GRAV = 6.6732E-8*ZR*ZT**2
      DT = 1.E-2
      DTH =DT/2.
      KOUNTT = 0
      KOUNTO = 0
C
      starl = 1.
      dist = 5.2
      alb=0.35
      teffs=5770.e0
      tj=106.e0
      radsun=1.
C
      UOUT = 19
      UCONT = 23
      UDUMP = 24
      UREAD = 31
      UCONV = 29
      UT = 7
      USTAB = 8
      UPTAB = 9
      UY = 21
      UDEN = 12
      UE = 17
      UFILE=35
      USTEP=20
C
      CALL SIGMA
