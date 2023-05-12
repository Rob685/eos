          AREA(I) = PI4*RADM(I)**2
          AREAO(I) = PI4*RADOM(I)**2
          BENCM(I) =  ( ENCM(I) + 1.D-10)
          BXM(I) = BENCM(I)**.666666666D0
          BXP(I) = BENCM(I+1)**.666666666D0
          BX(I) = 0.6D0*(BENCM(I+1)*BXP(I) - BENCM(I)*BXM(I))
     *        / (BENCM(I+1) - BENCM(I))
          GRM(I) = 1.D0- 4.4266D-3*GRAV*GM(I)/(RADM(I) + UNDFLW)
          GRP(I) = 1.D0- 4.4266D-3*GRAV*GM(I+1)/RADP(I)
          VOL(I) = PI43*(RADP(I)**3 - RADM(I)**3)
          FEXP(XX)=EXP(MAX(MIN(XX,40.D0),-40.D0))
          GEXP(XX)=EXP(-MAX(MIN(XX,40.D0),-40.D0))
          HEXP(XX)=1.D0-GEXP(XX)
          DLAM(XX,ZZ)=0.22745*(XX*(1.-YHEA/2))**(1./3.)*1.E6/ZZ
          QQ(XX,ZZ)=MIN(.977*(DLAM(XX,ZZ))**1.29,1.057*DLAM(XX,ZZ))
          SCR(XX,ZZ)=MIN(1.D3,FEXP(QQ(XX,ZZ)))
C  - DEUTERIUM BURNING
          T2(ZZ)=1.+1.12E-4*ZZ**(1./3.)+1.99E-6*ZZ**(2./3.)+1.56E-9*ZZ
          T1(XX,YY,ZZ)=XX*YY*EXP(-37.2*(1.D2/ZZ**(1./3.)-1.D0))*T2(ZZ)
C Note the factor 0.845 from Harris et al '83
          THERMO(XX,YY,ZZ)=0.845*(1.-YHEA)*9.8136E21/ZZ**(2./3.)
     *                    *T1(XX,YY,ZZ)*SCR(XX,ZZ)
          DRATE(XX,YY,ZZ)=-THERMO(XX,YY,ZZ)/(YY+UNDFLW)*1.886524E-19
C  - HYDROGEN BURNING
          HT2(ZZ)=1.+1.23E-4*ZZ**(1./3.)+1.09E-6*ZZ**(2./3.)+.938E-9*ZZ
          HT1(ZZ)=EXP(-33.8*(1.D2/ZZ**(1./3.)-1.D0))*HT2(ZZ)
C Note the factor of 0.907 from Harris et al '83
          HTHERM(XX,ZZ)=0.907*5.1734E5*XX*(1-YHEA)**2./ZZ**(2./3.)
     *                 *HT1(ZZ)*SCR(XX,ZZ)
C  - LITHIUM-7 BURNING
          T1L(XX,YY,ZZ)=XX*YY*EXP(-84.71*(1.D2/ZZ**(1./3.)-1.D0))*T2(ZZ)
          THERMOL(XX,YY,ZZ)=(1.-YHEA)/ZZ**(2./3.)
     *                    *T1L(XX,YY,ZZ)*SCR(XX,ZZ)
          DRATEL(XX,YY,ZZ)=-THERMOL(XX,YY,ZZ)/(YY+UNDFLW)*1.3067e-12
C  - LITHIUM-6
          T2L(XX,YY,ZZ)=XX*YY*EXP(-84.13*(1.D2/ZZ**(1./3.)-1.D0))*T2(ZZ)
          THEMOL(XX,YY,ZZ)=(1.-YHEA)/ZZ**(2./3.)
     *                    *T2L(XX,YY,ZZ)*SCR(XX,ZZ)
          DRAEL(XX,YY,ZZ)=-THEMOL(XX,YY,ZZ)/(YY+UNDFLW)*1.10305e-10
