
C
C      This routine performs initializations prior to use of
C      the lookup routine, LOOK:
C
       SUBROUTINE SETTABL
       IMPLICIT DOUBLE PRECISION (A-H,O-Z)
       COMMON/THERM/ SL(330,100),PL(330,100)
       COMMON/FIRST/ VEOS
       COMMON/FIRST2/ JFIRST
       COMMON/HELIUM/ YHEA
       COMMON/TABLE/ R1,R2,T1,T2,T12,T22,INDEX
       OPEN(8,FILE='stabnew.dat',STATUS='OLD')
       OPEN(9,FILE='ptabnew.dat',STATUS='OLD')
C
       READ(8,*) YHEA,INDEX,R1,R2,T1,T2,T12,T22
       DO 200 JR = 1,INDEX
       DO 150 JQS=1,10
       JL = 1 + (JQS-1)*10
       JU = JL + 9
       READ(8,130) (SL(JR,JQ),JQ=JL,JU)
130    FORMAT(10F8.5)
150    CONTINUE
       WRITE(*,*) '----------finished reading S table-----------'
C
       READ(9,*) YHEA,INDEX,R1,R2,T1,T2,T12,T22
       DO 300 JR=1,INDEX
       DO 160 JQP=1,10
       JL = 1 + (JQP-1)*10
       JU = JL + 9
       READ(9,130) (PL(JR,JQ),JQ=JL,JU)
160    CONTINUE
300    CONTINUE
       WRITE(*,*) '----------finished reading P table-----------'
C
       CLOSE(8)
       CLOSE(9)
C
C       Initialize EOS routine for cases off the table
C      (SUBSNEW must be linked):
       YY = YHEA
       JFIRST=1
       CALL SUBSNEW(1.D-3,1.D3,PDUM,SDUM,YY)
       JFIRST=0
       RETURN
       END
C
C
C  --------------------------------------------------
C
C     This program interpolates in the thermodynamic table
C       produced by FILTABL.
C
       SUBROUTINE LOOK(R,T,FP,FS)
       IMPLICIT DOUBLE PRECISION (A-H,O-Z)
       COMMON/THERM/ SL(330,100),PL(330,100)
       COMMON/FIRST/ VEOS
       COMMON/FIRST2/ JFIRST
       COMMON/HELIUM/ YHEA
       COMMON/TABLE/ R1,R2,T1,T2,T12,T22,INDEX
       IF(VEOS .EQ. 2.) THEN
        CALL SUBSOLD(R,T,FP,FS,YHEA)
        RETURN
       ELSE
       ENDIF
C       looks up S
C       then looks up P
C       Uses a six-pt. bivariate formula; see Abram. & Stegun, p. 882.
       RL = DLOG10(R)
       ALPHA=T1+(RL-R1)/(R2-R1)*(T12-T1)
       BETA=T2-T1+((T22-T12)-(T2-T1))*(RL-R1)/(R2-R1)
       QL = (DLOG10(T) - ALPHA)/BETA
       DELTA=(RL-R1)/(R2-R1)*FLOAT(INDEX)
       JR = 1 + IDINT(DELTA)
       JQ = 1 + IDINT(100.*QL)
       IF(JR.LT.2) GO TO 300
       IF(JR.GT.(INDEX-1)) GO TO 300
       IF(JQ.LT.2) GO TO 300
       IF(JQ.GT.99) GO TO 300
       P = DELTA - (JR-1)
       Q = 100.*QL - (JQ-1)
C        interpolate:
       FS = 0.5D0*Q*(Q-1.D0)*SL(JR,JQ-1)
     1  + 0.5D0*P*(P-1.D0)*SL(JR-1,JQ)
     2  + (1.D0+P*Q-P*P-Q*Q)*SL(JR,JQ)
     3  + 0.5D0*P*(P-2.D0*Q+1.D0)*SL(JR+1,JQ)
     4  + 0.5D0*Q*(Q-2.D0*P+1.D0)*SL(JR,JQ+1)
     5  + P*Q*SL(JR+1,JQ+1)
        FS = 10.D0**FS
C
C VEOS .EQ. 3.d0 means that the tables contain the MH EOS.
C It should be set up in bdtoolnew.dat.  VEOS .EQ. 1.d0 means
C that the SC EOS is in the tables.
C
       IF(VEOS .EQ. 3.d0)FS = (1.-yhea)*FS
C
       FP = 0.5D0*Q*(Q-1.D0)*PL(JR,JQ-1)
     1  + 0.5D0*P*(P-1.D0)*PL(JR-1,JQ)
     2  + (1.D0+P*Q-P*P-Q*Q)*PL(JR,JQ)
     3  + 0.5D0*P*(P-2.D0*Q+1.D0)*PL(JR+1,JQ)
     4  + 0.5D0*Q*(Q-2.D0*P+1.D0)*PL(JR,JQ+1)
     5  + P*Q*PL(JR+1,JQ+1)
        FP = 10.D0**FP
       IF(VEOS .EQ. 3.d0)FP = 1.D6*FP
      RETURN
C
C       off the table
300    CONTINUE
       if(jr .gt. 2) then
C        WRITE(*,*) ' off the table!  JR,JQ,rho: ',JR,JQ,R,T
C        WRITE(79,*) ' off the table!  JR,JQ,rho: ',JR,JQ,R,T
       else
       endif
       YY = YHEA
       CALL SUBSNEW(R,T,PRESS,ENTRP,YY)
CSC       FS = ENTRP
CSC       FP = PRESS
       FS = (1.-yhea)*ENTRP
       FP = 1.D6*PRESS
       RETURN
       END
C

      SUBROUTINE STATO
C
      INCLUDE 'bdcom.for'
C
      COMMON/CC/DPDR,DPDT,DSDT,DSDR
      INCLUDE 'bdfun.for'
      K=IZONE
      RHO=EXP(Q(K))
      PP=EXP(P(K))
 9    NITER=0
10    NITER=NITER+1
C
      if(niter .gt. 30 .and. ABS(DRXX).LT.2.d-3 .AND.
     *                              ABS(DTXX).LT.2.d-3) then
       go to 20
      else  if(niter .gt. 50 .and. ABS(DRXX).LT.2.d-2 .AND.
     *                              ABS(DTXX).LT.2.d-2) then
       go to 20
      endif
C
      RHO0=RHO
      CALL LOOK(RHO*1.1d0,T(K),P1,S1)
      CALL LOOK(RHO,T(K)*1.1d0,P2,S2)
      CALL LOOK(RHO,T(K),P0,S0)
      S1=RCON*S1
      S2=RCON*S2
      S0=RCON*S0
c
      DPDR=(P1-P0)/(.1d0*RHO)
      DPDT=(P2-P0)/(.1d0*T(K))
      DSDT=(S2-S0)/(.1d0*T(K))
      DSDR=(S1-S0)/(.1d0*RHO)
      DEN=DPDR*DSDT-DPDT*DSDR
      DRXX=((S0-S(K))*DPDT+(PP-P0)*DSDT)/(DEN*RHO)
      DTXX=((S(K)-S0)*DPDR-(PP-P0)*DSDR)/(DEN*T(K))
      XF=1.d0 ! -.5*(NITER/10-NITER/11)
      if(niter .gt. 10) xf=.5d0
      RHO=RHO*EXP(SIGN(min(1.d0,ABS(DRXX)),DRXX)*XF)
      T(K)=T(K)*EXP(SIGN(min(1.d0,ABS(DTXX)),DTXX)*XF)
      NUMBER(K)=NITER
      IF(ABS(DRXX).GT.1.d-3.OR.ABS(DTXX).GT.1.d-3) GOTO 10
 20   CONTINUE
      CV(K)=T(K)*DSDT
      CP(K)=T(K)*DEN/DPDR
      IF(DABS(DLOG10(RHO)) .LE. 0.15D0) THEN
       CV(K)=CV(1)
       CP(K)=CP(1)
       DQ(K)=DSDT*PP/(DEN*RHO0) ! 1./1.68
       TDPT(K)=T(K)*DPDT
      ELSE
       DQ(K)=DSDT*PP/(DEN*RHO0)
       TDPT(K)=T(K)*DPDT
      ENDIF
      Q(K)=log(RHO)
C
      RETURN
      END

