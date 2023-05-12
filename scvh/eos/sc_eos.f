C*
C     MAIN PROGRAM 
C*
C --  N.B. the lower-case bdcom.for
C
      INCLUDE 'bdcom.for'
      CHARACTER*1 ANSWER
      RCON=8.31434E7
C
      CALL SETTABL
C
      mode = 0
      write(6,*) 'What is the mode (=1:S,P; =0:T,R)?'
      read(5,*)  mode 
 1    continue
      if (mode .eq. 1) then
        WRITE(6,*) 'What are the Entropy and Pressure (in bars)?'
        read(5,*) aves,pnew
        NK5 = 10
        P(NK5)=DLOG(pnew*1.D6)
C Guesses
c       DENSE = 2.6878d-2
c       T(NK5)=4410.d0
        DENSE = 2.d-4
        T(NK5)=200.
C
        Q(NK5)=DLOG(DENSE)
        S(NK5)=AVES*RCON
        IZONE=NK5
        CALL STATO
        R(NK5)=EXP(Q(NK5))
        T102=T(NK5)
        write(6,*) 'S,P,R,T=',AVES,EXP(P(NK5))/1.d6,R(NK5),T102
      else if(mode .eq. 0) then
C
        WRITE(6,*) 'What are the density and temperature?'
        read(5,*) dense, tnew
        call LOOK(DENSE,TNEW,FP,FS)
        write(6,*) 'R,T,P,S=',DENSE,TNEW,FP/1.d6,FS
      endif
c
      go to 1
c
      STOP
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
c     P1=1.D6*P1
c     P2=1.D6*P2
c     P0=1.D6*P0
c     S1=(1.d0-YHEA)*RCON*S1
c     S2=(1.d0-YHEA)*RCON*S2
c     S0=(1.d0-YHEA)*RCON*S0
      S1=RCON*S1
      S2=RCON*S2
      S0=RCON*S0
c
c     write(6,*) niter,k
c     write(6,*) ' T,rho,s,s0=',T(K),RHO,S(K),S0
c     write(6,*) ' P,P0=',PP,P0
      DPDR=(P1-P0)/(.1d0*RHO)
      DPDT=(P2-P0)/(.1d0*T(K))
      DSDT=(S2-S0)/(.1d0*T(K))
      DSDR=(S1-S0)/(.1d0*RHO)
C      DEN=(P1-P0)*(S2-S0)-(S1-S0)*(P2-P0)
C      DRXX=.001*((S0-S(K))*(P2-P0)+(PP-P0)*(S2-S0))/DEN
C      DTXX=.001*((P0-PP)*(S1-S0)+(S(K)-S0)*(P1-P0))/DEN
C      DEN=DPDR*DSDT+(DPDT/RHO)**2
      DEN=DPDR*DSDT-DPDT*DSDR
      DRXX=((S0-S(K))*DPDT+(PP-P0)*DSDT)/(DEN*RHO)
C      DTXX=((S(K)-S0)*DPDR+(PP-P0)*DPDT/RHO**2)/(DEN*T(K))
      DTXX=((S(K)-S0)*DPDR-(PP-P0)*DSDR)/(DEN*T(K))
      XF=1.d0 ! -.5*(NITER/10-NITER/11)
      if(niter .gt. 10) xf=.5d0
      RHO=RHO*EXP(SIGN(min(1.d0,ABS(DRXX)),DRXX)*XF)
      T(K)=T(K)*EXP(SIGN(min(1.d0,ABS(DTXX)),DTXX)*XF)
      NUMBER(K)=NITER
      IF(ABS(DRXX).GT.1.d-3.OR.ABS(DTXX).GT.1.d-3) GOTO 10
C     CV(K)=1.d3*(E2-E(K))/T(K)
 20   CONTINUE
      CV(K)=T(K)*DSDT
      CP(K)=T(K)*DEN/DPDR
C     DQ(K)=.001d0*PP/(P1-P0+(P2-P0)**2/(RHO*(E2-E(K))))
      IF(DABS(DLOG10(RHO)) .LE. 0.15D0) THEN
       CV(K)=CV(1)
       CP(K)=CP(1)
       DQ(K)=DSDT*PP/(DEN*RHO0) ! 1./1.68
C       DQ(K)=PP/(RHO0*DPDR)
       TDPT(K)=T(K)*DPDT
      ELSE
C       DQ(K)=PP/(RHO0*DPDR+DPDT**2/(RHO0*DSDT))
       DQ(K)=DSDT*PP/(DEN*RHO0)
       TDPT(K)=T(K)*DPDT
      ENDIF
      Q(K)=log(RHO)
C
      RETURN
      END
C
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
c      OPEN(8,FILE='s25scy.dat',STATUS='OLD')
c      OPEN(9,FILE='p25scy.dat',STATUS='OLD')
c      OPEN(8,FILE='/home/burrows/cool/STAB025_extend.2.dat',STATUS='OLD')
c      OPEN(9,FILE='/home/burrows/cool/PTAB025_extend.2.dat',STATUS='OLD')
C
       VEOS = 1.d0
       READ(8,*) YHEA,INDEX,R1,R2,T1,T2,T12,T22
       DO 200 JR = 1,INDEX
       DO 150 JQS=1,10
       JL = 1 + (JQS-1)*10
       JU = JL + 9
       READ(8,130) (SL(JR,JQ),JQ=JL,JU)
130    FORMAT(10F8.5)
150    CONTINUE
200    CONTINUE
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
C       PRINT *, SL(182, 3)
C       PRINT *, SL(183, 3)
C       PRINT *, PL(182, 3)
C       PRINT *, PL(183, 4)
C
       CLOSE(8)
       CLOSE(9)
C       
       RETURN
       END
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
       IF(VEOS .EQ. 2.d0) THEN
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
       
       PRINT *, DELTA, IDINT(DELTA)
       
C       IDINT()
C        interpolate:

C	   PRINT *, SL(JR,JQ-1),SL(JR-1,JQ),SL(JR,JQ),SL(JR+1,JQ),SL(JR,JQ+1),SL(JR+1,JQ+1)

C	   PRINT *, SL(JR,JQ-1), 
	   
       FS = 0.5D0*Q*(Q-1.D0)*SL(JR,JQ-1)
     1  + 0.5D0*P*(P-1.D0)*SL(JR-1,JQ)
     2  + (1.D0+P*Q-P*P-Q*Q)*SL(JR,JQ)
     3  + 0.5D0*P*(P-2.D0*Q+1.D0)*SL(JR+1,JQ)
     4  + 0.5D0*Q*(Q-2.D0*P+1.D0)*SL(JR,JQ+1)
     5  + P*Q*SL(JR+1,JQ+1)
        FS = 10.D0**FS
C
       FP = 0.5D0*Q*(Q-1.D0)*PL(JR,JQ-1)
     1  + 0.5D0*P*(P-1.D0)*PL(JR-1,JQ)
     2  + (1.D0+P*Q-P*P-Q*Q)*PL(JR,JQ)
     3  + 0.5D0*P*(P-2.D0*Q+1.D0)*PL(JR+1,JQ)
     4  + 0.5D0*Q*(Q-2.D0*P+1.D0)*PL(JR,JQ+1)
     5  + P*Q*PL(JR+1,JQ+1)
        FP = 10.D0**FP
        
C        PRINT *, P, Q
        
C    	WRITE(*,*) 'P test, S test',FP,FS
C		PRINT *, FP, FS
      RETURN
C
C       off the table
300    CONTINUE
       if(jr .gt. 2) then
         WRITE(*,*) ' off the table!  JR,JQ,rho: ',JR,JQ,R,T
C        WRITE(79,*) ' off the table!  JR,JQ,rho: ',JR,JQ,R,T
       else
       endif
       RETURN
       END
C
      SUBROUTINE SUBSOLD(RHO,T,P,S,YYHE)
C
      implicit real*8(a-h,o-z)
C
      COMMON /B/ RCON,CFLUX
      COMMON /C/ PE,PTHERM
      COMMON/CC/DPDR,DPDT,DSDT,DSDR
      DATA TCON /40.55957/
C
      PE=9.9E12*RHO**(5./3.)
      PTHERM=RCON*RHO*T
      P=(PE+PTHERM)/1.D6
      E=(3./2.)*P/RHO
      S=LOG((T/TCON)**(1.5)/RHO)  *  (1.-YYHE)
      DSDT=(3./2.)*RCON/T
      DPDT=PTHERM/T
      DPDR=((5./3.)*PE+PTHERM)/RHO
      CV=3.*RCON/2.
C
      RETURN
      END
c
