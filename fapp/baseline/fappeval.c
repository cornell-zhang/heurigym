/*------------------------Definition des Include-------------------------*/
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdlib.h>


/*========================================================================*/
#define min(A,B) ((A<B) ? (A) : (B))
#define max(A,B) ((A>B) ? (A) : (B))

#define pair(A) (((A%2)==0) ? (1) : (0))
#define impair(A) (((A%2)!=0) ? (1) : (0))

/*========================================================================*/
#define	NB_MAX_TRAJETS		3000
#define NB_MAX_CONTRAINTES	100000
#define NB_MAX_DOMAINES		100
#define TAILLE_MAX_DOMAINE	1500

/*========================================================================*/
typedef struct{
	int	NUMERO;
	int	DOM_FREQ;
	int	DOM_POL;
	int	FREQ;
	int	POL;
	} TRAJET;

/*========================================================================*/

int main(int argc, char *argv[])
{
int	TAB_DOMAINES[ NB_MAX_DOMAINES ][ TAILLE_MAX_DOMAINE ];
TRAJET	TAB_TRAJETS[ NB_MAX_TRAJETS ];
int	TAB_NUMERO[ NB_MAX_TRAJETS ]; /* Donne la corresp. entre le numero de trajets et son indice dans TAB_TRAJETS */
int	NB_DOMAINES;
int	NB_TRAJETS;
int	NB_CONTRAINTES;

int	kmax;
int	TAB_VIOL[ 11 ];
int	VIOL_CI;
int	SOM_VIOL;

int	THETA;

FILE	*fich;
char	FICH_IN[50];
char	FICH_OUT[50];

int	IN_DOM;
char	TYPEREC[3];
int	Indice_trajet;
int	Indice_trajet1,Indice_trajet2;
int	i1,i2,i3;
char	X1,X2;
int	ii,jj;

/*====================================*/
/* INITIALISATION DES NOMS DE FICHIER */
/*====================================*/

sprintf(FICH_IN,"%s.in",argv[1]);
sprintf(FICH_OUT,"%s.out",argv[1]);

/*==============================*/
/* LECTURE DU FICHIER RESULTATS */
/*==============================*/

/* Ouverture du fichier resultats */
fich = fopen( FICH_OUT , "r" );
if ( fich == NULL )
  {
	printf( " Erreur : Fichier fich.out non trouve \n" );
	exit( 0 );
  }

/* Initialisation de TAB_NUMERO a (-1) */
for ( ii = 0 ; ii < NB_MAX_TRAJETS ; ii++ )
  {
	TAB_NUMERO[ ii ] = -1 ;
  }

/* La premiere ligne n'est pas traitee */
fscanf(fich,"%*[^\n]\n");

/* Lecture des allocations de frequence et de polarisation */
NB_TRAJETS = 0;
while ( fscanf( fich , "AL %5d %5d %2d\n" , &i1 , &i2 , &i3 ) != EOF )
  {
	TAB_TRAJETS[ NB_TRAJETS ].NUMERO = i1;
	TAB_NUMERO[ i1 ] = NB_TRAJETS;
	TAB_TRAJETS[ NB_TRAJETS ].FREQ = i2;
	TAB_TRAJETS[ NB_TRAJETS ].POL = i3;
	NB_TRAJETS++;
  }	

/* Fermeture du fichier resultats */
fclose( fich );

/*============================================*/
/* LECTURE DU FICHIER D'ENTREE ET TRAITEMENTS */
/*============================================*/

/* Initialisation diverses */
NB_DOMAINES = 0;
for ( ii = 0 ; ii < NB_MAX_DOMAINES ; ii++ )
  {
	TAB_DOMAINES[ ii ][ 0 ]= 0;
  }
kmax = 0;
for ( ii = 0 ; ii <= 10 ; ii++ )
  {
	TAB_VIOL[ ii ] = 0;
  }
VIOL_CI = 0;
SOM_VIOL = 0;
THETA = 0;

/* Ouverture du fichier d'entree */
fich = fopen( FICH_IN , "r" );
if ( fich == NULL )
  {
        printf( " Erreur : Fichier fich.in non trouve \n" );
        exit( 0 );
  }

/* Debut des traitements sur les donnees du probleme */
while ( fscanf( fich ,"%2s ", TYPEREC ) != EOF )
  {
	switch ( TYPEREC[ 1 ] )
	  {
		case 'M' :
		  {
			fscanf( fich ,"%5d %5d\n", &i1 , &i2 );
			NB_DOMAINES = max( i1 , NB_DOMAINES );
			TAB_DOMAINES[ i1 ][ 0 ]++;
			TAB_DOMAINES[ i1 ][ TAB_DOMAINES[ i1 ][ 0 ] ] = i2;
			break;
		  }

		case 'R' :
		  {
			fscanf( fich ,"%5d %5d %2d\n", &i1 , &i2 , &i3 );
			Indice_trajet = TAB_NUMERO[ i1 ];
			TAB_TRAJETS[ Indice_trajet ].DOM_FREQ = i2;
			TAB_TRAJETS[ Indice_trajet ].DOM_POL = i3;
			/* Verification de la validite de la frequence affectee */
			IN_DOM = 0;
			ii = 1;
			while ( ( IN_DOM == 0 ) && ( ii <= TAB_DOMAINES[ i2 ][ 0 ] ) )
			  {
				if ( TAB_TRAJETS[ Indice_trajet ].FREQ == TAB_DOMAINES[ i2 ][ ii ] )
				  {
					IN_DOM = 1;
				  }
				 else
				  {
					ii++;
				  }
			  }
			if ( IN_DOM == 0 )
			  {
				printf( "Erreur : Frequence du trajet %d en dehors de son domaine\n",i1);
			  }

			/* Verification de la validite de la polarisation affectee */
			if ( ( ( TAB_TRAJETS[ Indice_trajet ].DOM_POL == (-1) ) && 
					( TAB_TRAJETS[ Indice_trajet ].POL != ( -1 ) ) ) || 
				( ( TAB_TRAJETS[ Indice_trajet ].DOM_POL == 1 ) && 
					( TAB_TRAJETS[ Indice_trajet ].POL != 1 ) ) )
			  {
				printf( "Erreur : Polarisation du trajet %d en dehors de son domaine\n",i1);
			  }
			break;
		  }
		
		case 'I' :
		  {
			fscanf( fich ,"%5d %5d %1c %1c %5d\n", &i1 , &i2 , &X1 , &X2 , &i3);
			Indice_trajet1 = TAB_NUMERO[ i1 ];
			Indice_trajet2 = TAB_NUMERO[ i2 ];
			if ( X1 == 'F' )
			  {
				/* Contrainte sur les frequences */
				if ( X2 == 'E' )
				  {
					/* Contraintes d'egalite sur un ecart de frequence */
					if ( abs( TAB_TRAJETS[ Indice_trajet1 ].FREQ - TAB_TRAJETS[ Indice_trajet2 ].FREQ ) != i3 )
					  {
						VIOL_CI++;
						printf("Erreur : Contrainte d'egalite sur un ecart de frequence non satisfaite entre les trajets %d et %d\n",i1,i2);
					  }
				  }
				 else
				  {
					/* Contraintes d'inegalite sur un ecart de frequence */
					if ( abs( TAB_TRAJETS[ Indice_trajet1 ].FREQ - TAB_TRAJETS[ Indice_trajet2 ].FREQ ) == i3 )
					  {
						VIOL_CI++;
						printf("Erreur : Contrainte d'inegalite sur un ecart de frequence non satisfaite entre les trajets %d et %d\n",i1,i2);
					  }
				  }
			  }
			 else
			  {
				/* Contrainte sur les polarisations */	
				if ( X2 == 'E' )
				  {
					/* Polarisation egales ? */
					if ( TAB_TRAJETS[ Indice_trajet1 ].POL != TAB_TRAJETS[ Indice_trajet2 ].POL )
					  {
						VIOL_CI++;
						printf("Erreur : Contrainte de polarisations egales non satisfaite entre les trajets %d et %d\n",i1,i2);
					  }
				  }
				 else
				  {
					/* Polarisation croisees ? */
					if ( TAB_TRAJETS[ Indice_trajet1 ].POL == TAB_TRAJETS[ Indice_trajet2 ].POL )
					  {
						VIOL_CI++;
						printf("Erreur : Contrainte de polarisations croisees non satisfaite entre les trajets %d et %d\n",i1,i2);
					  }
				  }
			  }
			break;
		  }

		case 'E' :
		  {
			fscanf( fich ,"%5d %5d", &i1 , &i2 );
			Indice_trajet1 = TAB_NUMERO[ i1 ];
			Indice_trajet2 = TAB_NUMERO[ i2 ];
			if ( TAB_TRAJETS[ Indice_trajet1 ].POL !=  TAB_TRAJETS[ Indice_trajet2 ].POL )
			  {
				/* On ne prend pas en compte la contrainte car polarisations croisees */
				fscanf(fich,"%*[^\n]\n");
			  }
			 else
			  {
				THETA++;

				/* On analyse en detail la contrainte CEM entre i1 et 12 */
				for ( ii = 0 ; ii <= 10 ; ii++ )
				  {
					fscanf( fich ," %5d", &i3 );
					if ( abs( TAB_TRAJETS[ Indice_trajet1 ].FREQ - TAB_TRAJETS[ Indice_trajet2 ].FREQ ) < i3 )
					  {
						/* La contrainte de niveau ii n'est pas satisfaite */
						TAB_VIOL[ ii ]++;
						kmax = max( kmax , ii );
					  }
				  }
				fscanf( fich ,"\n");
			  }
			break;
		  }

		case 'D' :
		  {
			fscanf( fich ,"%5d %5d", &i1 , &i2 );
			Indice_trajet1 = TAB_NUMERO[ i1 ];
			Indice_trajet2 = TAB_NUMERO[ i2 ];
			if ( TAB_TRAJETS[ Indice_trajet1 ].POL ==  TAB_TRAJETS[ Indice_trajet2 ].POL )
			  {
				/* On ne prend pas en compte la contrainte car polarisations egales */
				fscanf(fich,"%*[^\n]\n");
			  }
			 else
			  {
				THETA++;

				/* On analyse en detail la contrainte CEM entre i1 et 12 */
				for ( ii = 0 ; ii <= 10 ; ii++ )
				  {
					fscanf( fich ," %5d", &i3 );
					if ( abs( TAB_TRAJETS[ Indice_trajet1 ].FREQ - TAB_TRAJETS[ Indice_trajet2 ].FREQ ) < i3 )
					  {
						/* La contrainte de niveau ii n'est pas satisfaite */
						TAB_VIOL[ ii ]++;
						kmax = max( kmax , ii );
					  }
				  }
				fscanf( fich ,"\n");
			  }
			break;
		  }
	  }
  }


/*=========================*/
/* AFFICHAGE DES RESULTATS */
/*=========================*/

printf("\n\n*****************************\n\n");
printf("RESULTS: (THETA = %d )\n\n",THETA);
printf("\tNumber of unsatisfied mandatory constraints: %d\n",VIOL_CI);
printf("\tMinimum relaxation level: k* = %d\n",kmax+1);
printf("\tNumber of violations per level: \n\t\t");
for ( ii = 0 ; ii <= 10 ; ii++ )
  {
	printf("%d ",TAB_VIOL[ ii ]);
	SOM_VIOL += TAB_VIOL[ ii ];
  }
printf("\n");
printf("\tNumber of violations at level %d (k*-1) = %d\n",kmax,TAB_VIOL[ kmax ]); 
printf("\tTotal number of violations for levels i < k*-1 = %d\n",SOM_VIOL - TAB_VIOL[ kmax ]);

printf("\n*****************************\n");
exit(0);
}
