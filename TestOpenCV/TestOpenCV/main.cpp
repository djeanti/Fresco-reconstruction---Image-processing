#include <cstdio>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <ctime> 

using namespace cv; 
using namespace std;

//variables globales :
string nomFichierFragments = "fragments_created.txt";//correspond au nom du fichier ou on stocke les informations du fragment pour reconstruire la fresque dans la fonction create_file

class Backproj {
    /*
        Cette classe calcul pour un fragment donn?la backprojection de la fresque associée
        On appelle la fonction new_hist() dès lors qu'on souhaite calculer un nouvel histogramme et la backprojection associée
        La backprojection de la fresque est toujours stockée dans la variable "backproj"
    */
public:
    Backproj(){};
protected:
    Mat src;//fresque
    Mat frag; // fragment
    Mat masque;//supprime le pic 0 au début de l'histogramme
    int bins;
    Mat hist;//histogramme courant
    Mat backproj;//backprojection courante

    //detection de points-clés (attributs nécessaires):
    Ptr<BRISK> detector;//le détecteur, celui qui détecte les points-clés de chaque image
    cv::BFMatcher bfmatcher;//le matcher, celui qui trouve les paires de points-clés entre kp_frag et kp_backproj
    vector<cv::DMatch> matches;//la vecteur contenant l'ensemble des distances trouvées par le matcher
    vector<KeyPoint> kp_backproj, kp_frag;//respectivement les points clés détéctés de la fresque (a laquelle on applique le masque de la backprojection) et du fragment
    cv::Mat descriptorsA, descriptorsB;//les descripteurs respectivement de la fresque et du fragment

    vector<DMatch> good_Matches;//vecteur contenant les distances filtrées correspondantes ?la position exacte du fragment sur la fresque
    vector<Point_< int >> dectedPoints;//vecteurs pour nous aider ?déterminer les coordonnées des fragments

public:
    //méthodes
    void init_fragment(Mat img_fresque_hsv, Mat img_frag_hsv, Mat _masque) 
    {
        //initialise l'ensemble des matrices source/destination nécessaires aux futurs calculs de l'histogramme et de la backprojection
        src = img_fresque_hsv;//fresque en HSV
        frag = img_frag_hsv;//fragment en HSV
        masque = _masque;//masque ?appliqu?pour le caclcul de l'histogramme du fragment (c'est une image ?niveau de gris)
        bins = 30;
        init_hist_tab();
    }

    void seuillage_backproj() 
    {
        Mat img;
        cv::threshold(backproj, img, 100, 255, cv::THRESH_BINARY);
        backproj = img;
    }

    void init_hist_tab()
    {//Détermine l'histogramme et la backprojection
        //paramètres :
        int channels[] = { 0,1,2 };
        int histSize[] = { bins,bins,bins };
        float hR0[] = { 0 , 180 };
        float hR12[] = { 0, 256 };
        const float* ranges[] = { hR0, hR12, hR12 };

        //calculs :
        calcHist(&frag, 1, channels, masque, hist, 3, histSize, ranges, true, false);
        calcBackProject(&src, 1, channels, hist, backproj, ranges, 1, true);//fresque complete et histogramme du fragment
        //seuillage_backproj(); // le seuillage reduit le nombre de points cl?qu'on détecte.
    }

    void show_bacjproj(string name="backproj") {
        imshow(name,backproj);
    }

    //trois méthodes pour savoir si on a bien détect?les keypoints ou non
    size_t get_kp_frag_sz() { return kp_frag.size(); }
    size_t get_kp_backproj_sz() { return kp_backproj.size(); }
    size_t get_good_Matches_sz() { return good_Matches.size(); }

    void keyPoint(Mat frag_gray, Mat fres_gray, bool show_all_matches=false)
    {//On récupère l'ensemble des points-clés dans kp_backproj et kp_frag respectivement pour l'image de la fresque et l'image du fragment
        detector = BRISK::create(20, 4, 1.0f);//les paramètres de cette méthode sont importants : on choisit d'utiliser 4 octaves et un treshold de 20, plus le treshold est petit et plus on détecte de points clés mais la précision diminue aussi (c'est-?dire qu'on risque de ne plus détecter correctement la zone sur la fresque ou se trouve le fragment si le treshold est trop élev?lors de l'étape du filtrage).

        //Points-clés de la fresque :
        detector->detect(fres_gray, kp_backproj, backproj);//on applique un masque en troisème paramètre pour réduire le nombre de points-clés, ce masque est la backprojection précédemment calculée
        detector->compute(fres_gray, kp_backproj, descriptorsA);
        if (kp_backproj.size() == 0) {//erreur
            return;
        }

        //Points-clés du fragment :
        detector->detect(frag_gray, kp_frag);
        detector->compute(frag_gray, kp_frag, descriptorsB);
        if (kp_frag.size()==0) {//erreur
            return;
        }

        //Le matcher :
        bfmatcher.create(NORM_HAMMING, true);
        bfmatcher.match(descriptorsB, descriptorsA, matches);//l'ordre dans lequel on écrit les arguments de cette méthode est très important. En effet, ici on cherche ?détecter les points-clés du fragment dans la fresque. L'inverse donne un résultat totalement différent et incorrect. 

        float mindist = 0, maxdist = 0;
        float var_min = 10000.0f;
        if (matches.size() == 0)//erreur
        {
            cout << "Pas de point cle qui match" << endl;
            return;
        }
        else {
            //on détermine deux distances seuils pour nous permettre de filtrer plus efficacement les points-clés : mindist et maxdist
            vector<float> ranged_dist;
            for (int i = 0; i < matches.size(); i++) {
                ranged_dist.push_back(matches[i].distance);
            }
            sort(matches.begin(), matches.end());
            sort(ranged_dist.begin(), ranged_dist.end());
            mindist = ranged_dist.front();
            maxdist = ranged_dist.back();
        }

        //A present, on récupère les paires de points-clés correspondant ?la zone ou le fragment se trouve réellement sur la fresque, on stocke les distances correspondantes dans la matrice good_Matches
        int count = 0;
        for (int j = 0; j < matches.size(); j++) {
            int seuil_dist = MAX(frag_gray.rows, frag_gray.cols);
            int x_pos_init = matches.size() == 0 ? 0 : (int)kp_backproj[matches[0].trainIdx].pt.x;
            int y_pos_init = matches.size() == 0 ? 0 : (int)kp_backproj[matches[0].trainIdx].pt.y;
            if (matches[j].distance <= maxdist && abs(kp_backproj[matches[j].trainIdx].pt.x - x_pos_init) < seuil_dist && abs(kp_backproj[matches[j].trainIdx].pt.y - y_pos_init) < seuil_dist) {
                good_Matches.push_back(matches[j]);
                dectedPoints.push_back(kp_backproj[matches[j].trainIdx].pt);//on construit dectedPoints, un vecteur qui nous aidera ?déterminer les coordoonnées de la position fine du fragment par la suite
                count++;
            }
            if (count >= 3)
                break;
        }
        if (good_Matches.size() == 0)//on a pas trouver le fragment sur la fresque en filtrant sur les distances
        {
            return;
        }

        //pour afficher les matches entre les points-clés du fragment et ceux de la fresque :
        if (show_all_matches) {
            cv::Mat all_matches;
            cv::drawMatches(frag_gray, kp_frag, fres_gray, kp_backproj,
                good_Matches, all_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            cv::namedWindow("BRISK Filtered Matches", WINDOW_NORMAL);
            cv::imshow("BRISK Filtered Matches", all_matches);
        }

    }

    float rotate_frag(Mat img_frag){//permet de déterminer la rotation d'un fragment
        //nous n'avons pas pu finir d'implémenter cette méthode c'est pourquoi elle renvoit une valeur fausse : 0.00f
        //l'idée aurait ét?d'utiliser la matrice de rotation (sa formule mathématique) et les coordonnées 2D de la position fine du fragment pour en déduire la rotation de ce dernier.
        int w = img_frag.rows, h = img_frag.cols;
        
        float angle = 0.00f;

        return angle;
    }

    bool checkGoodMatchesSZ() { return (good_Matches.size() > 0); }//permet de s'assurer que nous avons détect?au moins une paire de points-clés dans la zone de la fresque ou le fragment est cens?être.

    Point getCoordFrag() {
        //renvoit la position fine du fragment sur la fresque :
        //récupère les coordonnées 2D de l'un des points-clés sur la fresque (correspondant ?l'endroit ou le fragment est cens?se trouver)
        //déduit ?partir de ces coordonnées la position fine du fragment sur la fresque (le coin supérieur gauche)
        Point_< int > frag_pt_in_fres;
        int sumX = 0, sumY = 0;
        for (int i = 0; i < dectedPoints.size(); i++) {
            sumX += dectedPoints[i].x;
            sumY += dectedPoints[i].y;
        }
        frag_pt_in_fres.x = (int)(sumX / dectedPoints.size());
        frag_pt_in_fres.y = (int)(sumY / dectedPoints.size());
        return frag_pt_in_fres;
    }

    //méthodes statiques pour reconstruire la fresque :
    void static getFragName(int i, string& name);
    int static getFrag(string name, Mat& frag_rgb);
    void static addLineFile(Point p, float angle, int numFrag);
    void static create_file(Mat fresque_rgb, int num_start = 0, int num_stop = 10);
    void static test_frag(int j);
    size_t static show_keypoint_frag(Backproj b, int num);
    void static placeFragment(Mat src_rgb, string nomFichier);
    void static clear_file(string name);
};

int main()
{
    // lire un fichier en tant qu'image
    Mat fresque_rgb = imread("./fresque0.ppm", IMREAD_COLOR);//fresque

    // afficher une erreur si le fichier de l'image n'existe pas
    if (fresque_rgb.empty()) {
        cout << "Image fresque non trouvee" << endl;
        return -1;
    }

    /*********************Debut de coder***********************/
    int TEST = 2;//variable a changer selon le test ?effectuer

    switch(TEST) {
        case 1://Pour tester fragments.txt : 
        {
            Backproj::placeFragment(fresque_rgb, "fragments.txt");//avec le fichier fournit (fragments.txt) cela fonctionne
            break;
        }
        case 2: //Pour tester notre algorithme :
        {
            Backproj::create_file(fresque_rgb,0,2);
            Backproj::placeFragment(fresque_rgb, nomFichierFragments);//ne donne pas un affichage correct vu que la methode censée calculer la rotation d'un fragment n'est pas fonctionnelle
            break;
        }
        case 3://Pour tester un fragment en particulier :
        {
            Backproj::test_frag(0);//i=numéro du fragment ?tester.
            break;
        }
        default:break;
    }

    /*********************Fin de coder*************************/


    // attendre qu'un bouton soit appuye pour arreter
    waitKey(0);
    return 0;
}

void Backproj::getFragName(int i, string& name) {//Pour recuperer le nom du fragment en fonction de son numéro (meme principe utilis?dans la fonction placeFragment) 
    name = "./frag_eroded/frag_eroded_";
    stringstream convert_int_to_string;//cette chaine a la particularite de pouvoir convertir un int en format string
    string num_str;

    convert_int_to_string << i; convert_int_to_string >> num_str;
    for (int i = 0; i < 4 - num_str.size(); i++) {
        name += '0';
    }
    name += num_str + ".ppm";
}

int Backproj::getFrag(string name, Mat& frag_rgb) {//recupere l'image du fragment associe ?un nom name
    frag_rgb = imread(name, IMREAD_COLOR);
    // afficher une erreur si le fichier de l'image n'existe pas
    if (frag_rgb.empty()) {
        cout << "Image fragment non trouvee" << endl;
        return -1;
    }
    return 0;
}

void Backproj::clear_file(string name) {//supprime le contenu d'un fichier
    ofstream file(name);
}

void Backproj::addLineFile(Point p, float angle, int numFrag) {//Permet d'ajouter une ligne ?la fin du fichier nomFichierFragments
    string nomFichier = nomFichierFragments;
    ofstream file(nomFichier, std::ios::app);//std::ios::app : pour ne pas effacer le contenu du fichier a chaque fois qu'on ouvre ce fichier.
    stringstream convert_stream; string ligne;
    if(file) 
    {
        char tab[16];
        sprintf_s(tab,"%0.3f",angle);//on ne veut que trois décimales apres la virgule
        ligne = to_string(numFrag) + " " + to_string(p.x) + " " + to_string(p.y) + " " + tab + "\n";//construction du format de la ligne a ecrire dans le fichier (meme format que pour fragments.txt)
        file << ligne;
    }
    else {
        cout << "Impossible d'ecrire dans le fichier pour le fragment " << numFrag << endl;
    }
}

void Backproj::create_file(Mat fresque_rgb, int num_start, int num_stop) {//num_start et num_stop sont les numeros des fragments pour lesquels on applique l'algorithme, par defaut on test les 30 premiers fragments
    //la fonction create_file construit le fichier nomFichierFragments contenant toutes les informations nécessaires ?la reconstruction de la fresque

    Backproj::clear_file(nomFichierFragments);//on supprime le contenu de nomFichierFragments avant d'écrire dedans ( au cas ou il y aurait déj?quelque chose de rédig?dedans )

    float angle = 0.0f; Point p;
    string frag_name;
    Mat frag_rgb;
    float temps_backproj=0.00f;//pour le calcul du temps d'execution moyen de la fonction keypoint
    float temps_keypoints=0.00f;//pour le calcul du temps d'execution moyen de la fonction keypoint
    float temps_exec_frag=0.00f; //pour le calcul du temps d'execution moyen pour une iteration du while
    float temps_exec_total=0.00f;//temps pour reconstruire la fresque avec num_stop - num_start + 1 fragments
    int i;

    clock_t tfrag_begin = clock();

    //protection pour rester entre 0 et 314 (inclus):
    if (num_stop >= 315) num_stop = 314;
    if (num_start < 0) num_start = 0;

    for (i = num_start; i <= num_stop; i++) {//pour 1 seul fragment seulement
        getFragName(i, frag_name);
        if (getFrag(frag_name, frag_rgb)==0) {//on récupère le fragment associ?
            Mat frag_hsv, fresque_hsv, masque, frag_gray, fres_gray;
            cvtColor(frag_rgb, masque, COLOR_BGR2GRAY);
            cvtColor(fresque_rgb, fresque_hsv, COLOR_BGR2HSV);
            cvtColor(frag_rgb, frag_hsv, COLOR_BGR2HSV);

             //backprojection :
            Backproj b;
            clock_t tBackProj_begin = clock();
            b.init_fragment(fresque_hsv, frag_hsv, masque);//on calcul la backprojection du fragment
            clock_t tBackProj_end = clock();
            temps_backproj += (float)(tBackProj_end - tBackProj_begin) / CLOCKS_PER_SEC;

            //keypoint :
            cvtColor(frag_rgb, frag_gray, COLOR_BGR2GRAY);
            cvtColor(fresque_rgb, fres_gray, COLOR_BGR2GRAY);

            clock_t tKP_begin = clock();
            b.keyPoint(frag_gray, fres_gray);//détection de points-clés
            clock_t tKP_end = clock();
            temps_keypoints += (float)(tKP_end - tKP_begin) / CLOCKS_PER_SEC;

           if (b.checkGoodMatchesSZ())//si on trouve des points clés, on continue
            {
                angle = b.rotate_frag(frag_gray);//renvoit 0.00f car nous ne l'avons pas implémenter
                Point p = b.getCoordFrag();
                //on stocke les elements trouvés dans le fichier a la ligne i :
                addLineFile(p,angle,i);//on écrit ?la fin du fichier
            }
            
            clock_t tfrag_end = clock();
            temps_exec_frag += (float)(tfrag_end - tfrag_begin)/CLOCKS_PER_SEC;
            temps_exec_total = temps_exec_frag;
        }
    }
    temps_exec_frag /= (float)i;//temps moyen d'execution d'un fragment
    temps_keypoints /= (float)i;//temps moyen d'execution de la fonction keypoint
    temps_backproj /= (float)i;//temps moyen d'execution du clacul de la backprojection
    cout << "Temps d'execution pour 1 fragment en moyenne = " << temps_exec_frag << " sec" << endl;
    cout << "Temps d'execution pour la fonction keypoint en moyenne = " << temps_keypoints << " sec" << endl;
    cout << "Temps d'execution pour le calcul de la backprojection en moyenne = " << temps_backproj << " sec" << endl;
    cout << "Temps d'execution pour construire le fichier = " << temps_exec_total << " sec" << endl;
}

//Tester un fragment en particulier :
void Backproj::test_frag(int j) {
    Backproj b;
    vector<int> no_kp_frag_tab;
    vector<int> no_good_matches_tab;
    if (j>=315 || j<0) {
        cout << "Le numéro du fragment doit être compris appartenir a l'intervalle [0,314]" << endl;
        return;
    }

    int i=j;
    try {
        for (i = j; i < j+1; i++) {
            if (show_keypoint_frag(b, i)==0)
            {
                cout << "Aucun point-cl?trouv?pour ce fragment" << endl;
            }
        }
    }
    catch (Exception e) {
        cout << e.what() << endl;
    }
}

size_t Backproj::show_keypoint_frag(Backproj b, int num) {//fonction créé pour le debuggage
    //elle permet d'afficher les matches des points-clés entre la fresque et le fragment du fragment numéro i 
    //pour la valeur de retour : on renvoit le nombre de points cl?du fragment détéct?
    Mat fresque_rgb = imread("./fresque0.ppm", IMREAD_COLOR);
    if (fresque_rgb.empty()) {
        cout << "Image fresque non trouvee" << endl;
        return -1;
    }
    string nomFrag; getFragName(num, nomFrag);
    Mat frag_rgb = imread(nomFrag, IMREAD_COLOR); // choissisez le bon chemin de votre image
    if (frag_rgb.empty()) {
        cout << "Image fragment non trouvee" << endl;
        return -1;
    }

    Mat frag_hsv, fresque_hsv, masque, frag_gray, fres_gray;
    cvtColor(frag_rgb, masque, COLOR_BGR2GRAY);
    cvtColor(fresque_rgb, fresque_hsv, COLOR_BGR2HSV);
    cvtColor(frag_rgb, frag_hsv, COLOR_BGR2HSV);
    cvtColor(frag_rgb, frag_gray, COLOR_BGR2GRAY);
    cvtColor(fresque_rgb, fres_gray, COLOR_BGR2GRAY);

    //imshow("fragment", frag_gray);     waitKey(0);

    b.init_fragment(fresque_hsv, frag_hsv, masque);//on calcul la backprojection du fragment
    b.keyPoint(frag_gray, fres_gray, true);//détection de points-clés
    return b.get_kp_frag_sz();
}

//Pour reconstruire la fresque
void Backproj::placeFragment(Mat src_rgb, string nomFichier) //fragments.txt est le fichier fournit en TP contenant l'information des 315 fragments
/*
    Cette méthode statique permet de reconstruire la fresque complète ?partir d'un fichier texte contenant toutes les informations nécessaires et respectant un format précis.
    Chaque ligne a le format suivant : "numero_fragment position_Y position_X rotation" sachant que (position_X,position_Y) est la coordonnée 2D spatiale de la position fine du fragment sur la fresque
*/
{
    float temps_exec_total = 0.00f;//temps total pour reconstruire la fresque a partir du fichier donn?
    clock_t texec_begin = clock();
    clock_t texec_stop;

    unsigned int w = src_rgb.rows;
    unsigned int h = src_rgb.cols;
    if (w <= 0 || h <= 0) {
        return; //impossible de construire la fresque si le fichier source est incorrect.
    }
    Mat mat = src_rgb;
    for (unsigned int i = 0; i < w; i++)
    {
        for (unsigned int j = 0; j < h; j++)
        {
            mat.at<Vec3b>(i, j)[0] = mat.at<Vec3b>(i, j)[1] = mat.at<Vec3b>(i, j)[2] = 0;
        }
    }

    fstream file(nomFichier, ios::in);
    if (file) //dans un premier temps, on lit la totalit?du fichier pour stocker les informations nécessaires dans les variables déclarés ci-après.
    {
        //variables pour stocker les informations du fichier :
        string num;//numero du fragment
        int x = -1, y = -1;//position fine sur la fresque
        float rotation = 0.0f;
        string ligne;//pour parcourir le fichier

        //vecteurs (pour stocker les informations du fichier et les utiliser apres) :
        vector<Point> tab_coord;
        vector<string> tab_names;
        vector<float> tab_rotations;
        stringstream string_to_var;
        //dans le while suivant on lit le fichier jsuq'au bout
        while (getline(file, ligne)) {
            //on recupere les parametres :
            string_to_var << ligne;
            string_to_var >> num;
            string_to_var >> x;
            string_to_var >> y;
            string_to_var >> rotation;
            string_to_var.clear();//on se prepare a recevoir la prochaine ligne

            //technique pour déterminer le nom du fragment en utilisant uniquement son numéro
            string frag_name = "./frag_eroded/frag_eroded_";//debut de nom similaire ?tous les fragments
            for (int i = 0; i < 4 - num.size(); i++) {//Ex : si le numéro est 54 on rajoute deux '0' avant pour obtenir 0054
                frag_name += '0';
            }
            frag_name += num + ".ppm";

            tab_coord.push_back(Point(y, x));
            tab_names.push_back(frag_name);
            tab_rotations.push_back(rotation);
        }

        for (int num_frag = 0; num_frag < tab_names.size(); num_frag++)//a present on construit la fresque
        {
            //on recupere l'image du fragment
            Mat img = imread(tab_names.at(num_frag), IMREAD_COLOR);
            if (img.empty())
            {
                cout << "Image " << tab_names.at(num_frag) << " non trouvee" << endl;
                return;
            }
            unsigned int w_frag = img.rows, h_frag = img.cols;

            //on tourne le fragment avant de le placer dans la fresque connaissant la rotation :
            Mat img_rotated = getRotationMatrix2D(Point(w_frag / 2, h_frag / 2), (tab_rotations.at(num_frag)), 1);//affine transformation matrix for 2D rotation//
            warpAffine(img, img_rotated, img_rotated, img.size());//applying affine transformation//

            //on recupere la position fine du fragment (coin supérieur gauche) connaissant la position du milieu du fragment sur la fresque
            Point p = tab_coord.at(num_frag);// coord du milieu du fragment dans la fresque
            unsigned int top_left_x, top_left_y;
            top_left_x = p.x - (w_frag / 2);
            top_left_y = p.y - (h_frag / 2);

            //on construit l'image mat (=la fresque) :
            unsigned int ii = 0; unsigned int jj = 0;
            for (unsigned int i = top_left_x; i < top_left_x + w_frag; i++)
            {
                for (unsigned int j = top_left_y; j < top_left_y + h_frag; j++)
                {
                    if (i < w && j < h) //protection
                    {
                        ii = i - top_left_x;
                        jj = j - top_left_y;

                        if (ii < w_frag && jj < h_frag) //protection
                        {
                            for (unsigned int k = 0; k < 3; k++)
                            {
                                if (img_rotated.at<Vec3b>(ii, jj)[k] != 0) {
                                    try//sans le try, dès qu'il est impossible de travailler avec un pixel, on a une erreur d'execution donc on l'utilise pour ne pas arrêter de construire mat juste ?cause de certains pixels
                                    {
                                        mat.at<Vec3b>(i, j)[k] = (unsigned int)img_rotated.at<Vec3b>(ii, jj)[k];
                                    }
                                    catch (Exception e) {//on ne fait rien si on a une exception
                                        //cout << e.what() << endl;
                                    }
                                }
                            }//fin for k
                        }
                    }//fin if
                }//fin for j
            }//fin for j

        }//fin du for num_frag

    }//fin if
    texec_stop = clock();
    temps_exec_total = (float)(texec_stop - texec_begin)/CLOCKS_PER_SEC;
    cout << "Temps d'execution total pour reconstruire la fresque = " << temps_exec_total << " sec" << endl;
    imshow("fresque reconstruite", mat);//on affiche le résultat final
}