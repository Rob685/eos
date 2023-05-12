      PARAMETER  (MN = 202, MNP = MN + 1)
      implicit real*8(a-h,o-z)
      DIMENSION STOUT(10),STPLOT(15)
      COMMON /IOZ/      IMODEL,KOUNTT,KOUNTO,STOUT,STPLOT,
     *                  UCONT,UDUMP,UOUT,UPLOT,UREAD,UFILE,
     *                  UT,UY,ULUM,USTEP,
     *                  UDEN,UE,UCONV,LPOUT,LFILL,
     *                  LACRET,LDUMP, LPRINT, LREAD, CHECK,LCONT,
     *                  LSTART,STATIC,LCORR,LCONV,LBURN,LTRANS,
     *                   LTEST
      COMMON /IOZ2/
     *                  TIMEP,TIMEO,
     *                  DTIMEP,DTIMEO,STEPP,STEPO
      COMMON /NORMC/    ZE, ZM, ZP, ZR, ZRAD, ZRAD3, ZT, ZV,
     *                  ACONZ1,ACONZ2,ACONPS,ACONP1,ACONP,ACONPP
      COMMON /DUMMY/    Z1(MNP),Z2(MNP),MODE,
     *                  NUMBER(MN),NITERT,IDTY,IDTT
      COMMON /FIRST/    VEOS
      COMMON /FIRST2/   JFIRST
      COMMON /HELIUM/   YHEA
      COMMON /HYDRO1/   DT, DTO, DTC, DTH, FRACT,
     *                  GRAV,MASS,XMAS2,
     *                  DRM(MNP),XRM(MNP),RM(MNP),GM(MNP),
     *                  M(MN),ENCM(MNP),ENCMO(MNP),
     *                  N, NM, NO, NP, NOLD, ITCOR,
     *                  ISTEP, MAXSTP, MINSTP
      COMMON /HYDRO2 /  OVRFLW,
     *                  PI, PI4, PI43,
     *                  R(MN), RO(MN), 
     *                  RAD(MN), RADO(MN), RADM(MNP), RADOM(MNP),
     *                  TIME, TIMEF, TIMEI,
     *                  UNDFLW,
     *                  X(MN), XM(MNP)
      COMMON /FRA/      YHE(MN),YHEO(MN),
     *                  H2(MN),XA(MN),XAO(MN),ZH(MN)
      COMMON /THERM2/   SPK,AVES,ENT(MN),
     *                  S(MN),SO(MN),T(MN),TO(MN),
     *                  PRESS(MN),
     *                  GAMMA(MN),TDPT(MN),DVT(MN),GAM
      COMMON /FLUX/     ACCEL(MN),TEFF,TEFFO,TC,AJL,BJL,CJL,
     *                  TCN,BFMU,FMU(MN),BFMUO,FMUO(MN),
     *                  LIMM(MN),LUMM(MN),KAPPA(MN),
     *                  A1(MN), B1(MN), C1(MN), D1(MN),
     *                  CM(MN),MAG
      COMMON /FLUX2/    SN(MN),DSNE(MN),DSNN(MN),
     *                  LUMMO(MN),
     *                  CE(MN),DYHE(MN),
     *                  SE(MN),
     *                  DELT(MN),CV(MN),CP(MN),DTAU(MN),
     *                  TAU(MN),BFLX,FLX(MN),DTX(MN),DTXO(MN),
     *                  ZN(MN),ZEE(MN),DMUE(MN),MFPM(MN)
      COMMON /A/        Q(MNP),P(MNP),DP(MNP),DR(MNP),DQ(MNP),
     *                  E(MNP),RADL(MNP),FY(MNP),PSI(MNP),W(MNP),
     *                  WB(MNP),XB(MNP),Y(MNP),YB(MNP),Z(MNP),
     *                  ZB(MNP),G(MNP),A(MNP)
      COMMON /B/        RCON,CFLUX
      COMMON /TOTAL/    ENERGY,ENERGP,TBEG,SHINE,SHAVE,ILOOP,LOOP
      COMMON /ACCR/     POUT,PBOUND,AMASS,
     *                  MDOT,DGM0,DELTAP,BMASS,
     *                  RHOA,PA,EA,SA,VELA,ACMAX,TBEGA,
     *                  CQ(MN),SLOPE,NMASS,AMETH,KZONE,NOBOND
      COMMON /CONVE/    DC(MN),VCON(MN),CCON(MN),
     *                  DTCON,
     *                  A1E(MN),B1E(MN),C1E(MN),D1E(MN),CEC(MN),
     *                  SCALE(MN),ICON(MN),CMETH
      COMMON /C/        PE,PTHERM
      COMMON /ZONE/     IZONE
      COMMON /TBURN/    HDBI(MN),TLUMM,RATIO,DTBURN,SCREEN(MN),HB,
     *                  TSUM,BURNDS,DELS,DT10S,DTET10,
     *                  YD(MN),YDO(MN),YDI,DYD(MN),ALPHAL,
     *                  YL6(MN),YL6O(MN),YLI6,DYL6(MN),SNL6(MN),
     *                  DSNL6(MN),
     *                  YL7(MN),YL7O(MN),YLI7,DYL7(MN),SNL7(MN),
     *                  DSNL7(MN)
      common /radia/    starl,dist,alb,teffs,tj,radsun,xlummp
      COMMON /ATMOSP/   ATFILE
C
      EQUIVALENCE  (RADP, RADM(2) ),(RADOP, RADOM(2) ),(XP, XM(2) )
      REAL*8  MDOT, MASS, NMASS, M, RADP(MN), RADOP(MN), XP(MN)
      REAL*8 LUMM,LUMMO,LIMM
      INTEGER  UCONT,UDUMP,UOUT,UOUTB,UPLOT,UREAD,UT,UY,ULUM
      INTEGER UDEN,UE,UCONV,UFILE,USTEP
      INTEGER STOUT,STPLOT,AMETH,CMETH
      LOGICAL  LDUMP,LPRINT,LREAD,CHECK,LBURN,LACRET,LTEST
      LOGICAL  STATIC,LSTART,LCORR,LCONV,LTRANS,LPOUT,LFILL,LCONT
      REAL*8  MFPM,KAPPA
      CHARACTER*2 MAG,GAM
      CHARACTER*15 ATFILE
