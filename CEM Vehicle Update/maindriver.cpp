
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <chrono>
// #include <Eigen/Core>
#include <numeric> //iota
#include <algorithm> //min
#include <fstream>

#include <algorithm> // clamp

//const float CAR_LENGTH = 1.0f;

//const float DT = .02f;

const float DT = .1f;//.02;
const float CAR_LENGTH = 4.7;//1
const float SIM_EXTENT = 50;//2;

const bool LOAD_PARAMETERS = false;


const bool RANDOM_START = true;
const int IN_SIZE = 3;
const int OUT_SIZE = 4; // control mean and standard deviation
const int HIDDEN_SIZE = 200; //3

const int TRAJECTORY_SAMPLES = 150;//350;//90;

const bool USE_STOCHASTIC_SAMPLING = false;//true;
const bool CLIP_CONTROL = true;
const int CEM_ITERATIONS = 3500;//3500
const int CEM_BATCH_SIZE = 100;//100
const float CEM_ELITE_FRAC = .1f;
const float CEM_INIT_STDDEV = 1.0f;
const float CEM_NOISE_FACTOR = 0.005f;
const bool CEM_ADD_NOISE = true; // 0 is diminishing, 1 is constant
const int EVALUATION_SAMPLES = 1;//15;

const bool RUN_PARALLEL = true;
#define SAVE_OUTPUT

// Eigen::Vector3f desired_state(0,1,0);

// const Eigen::Vector3f GOAL_STATE( 0.198049f, -0.0234648f, -0.262067f );
//const Eigen::Vector3f GOAL_STATE( 2.821836f, -0.344624f, -.3841286f );
const Eigen::Vector3f GOAL_STATE(25,0,0);


using v3FnCall = Eigen::Vector3f (*)(int args);

void FixedUpdate( Eigen::Vector3f &state, Eigen::Vector2f control, float dt);



Eigen::Vector3f CarDynamics( Eigen::Vector3f state, Eigen::Vector2f control, float carLength )
{
    return Eigen::Vector3f( control[0] * cosf(state[2]), 
                            control[0] * sinf(state[2]), 
                            control[0] / carLength * tanf(control[1])   );
}

Eigen::Vector3f DDDynamics( Eigen::Vector3f state, Eigen::Vector2f control, float carLength )
{
    return Eigen::Vector3f( control[0] * cosf(state[2]), 
                            control[0] * sinf(state[2]), 
                            control[1] );
}

Eigen::Vector3f GenFeature(Eigen::Vector3f x, Eigen::Vector3f g)
{
    Eigen::Vector2f diff = g.head<2>() - x.head<2>();
    float rotdifx = cos(-x[2]) * diff[0] - sin(-x[2]) * diff[1];
    float rotdify = sin(-x[2]) * diff[0] + cos(-x[2]) * diff[1];

    return Eigen::Vector3f(rotdifx,rotdify,x[2]);
}

float GetReward(Eigen::VectorXf state, Eigen::VectorXf control, Eigen::VectorXf goal, float dt)
{
    //Eigen::Vector3f displacement = state - GOAL_STATE;
    //float dist = displacement.norm();
    //float temp = 1.0/(dist+.1f) - control.squaredNorm()*dt*.01f;
    float penalty = 0;
    //if (abs(control[0]) > 40)
    //    penalty += 0;//100;
    //if (abs(control[1]) > 3.14159)
    //    penalty += 100;

    Eigen::VectorXf feat = GenFeature(state,GOAL_STATE); 
    
    penalty += abs(feat[1]);

    //penalty += control.head<2>().squaredNorm()*dt*.01f;

    Eigen::Vector3f x = state.head<3>();
    //for (int i = 0; i < 5; ++i)
    //{
    //    FixedUpdate(x,control,dt);
    //}

    Eigen::Vector2f displacement = x.head<2>() - goal.head<2>();
    float dist = displacement.norm();
    //return 1.0/(dist+.1f) - penalty - control.head<2>().squaredNorm()*dt*.01f;
    return 1.0/(dist+.1f) - penalty; 
}

float GetRewardOld(Eigen::VectorXf state, Eigen::VectorXf control, Eigen::VectorXf goal, float dt)
{
    //Eigen::Vector3f displacement = state - GOAL_STATE;
    //float dist = displacement.norm();
    //float temp = 1.0/(dist+.1f) - control.squaredNorm()*dt*.01f;

    Eigen::Vector2f displacement = state.head<2>() - goal.head<2>();
    float dist = displacement.norm();
    return 1.0/(dist+.1f) - control.squaredNorm()*dt*.01f;
}


namespace Eigen{
template<class Matrix>
void write_binary(const char* filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}
template<class Matrix>
void read_binary(const char* filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
}
} // Eigen::

void SaveFile(std::string filename, Eigen::VectorXf v)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << v << std::endl;
    }
    return;
}

//Eigen::MatrixXf ReadFile(const char *filename)
//{
//    int cols = 0, rows = 0;
//    double buff[1000];
//
//    // Read numbers from file into buffer.
//    std::ifstream infile;
//    infile.open(filename);
//    while (! infile.eof())
//        {
//        std::string line;
//        std::getline(infile, line);
//
//        int temp_cols = 0;
//        std::stringstream stream(line);
//        while(! stream.eof())
//            stream >> buff[cols*rows+temp_cols++];
//
//        if (temp_cols == 0)
//            continue;
//
//        if (cols == 0)
//            cols = temp_cols;
//
//        rows++;
//        }
//
//    infile.close();
//
//    rows--;
//
//    // Populate matrix with numbers.
//    Eigen::MatrixXf result(rows,cols);
//    for (int i = 0; i < rows; i++)
//        for (int j = 0; j < cols; j++)
//            result(i,j) = buff[ cols*i+j ];
//
//    return result;
//};

Eigen::MatrixXf ReadFile(const char *filename)
{
    //std::ofstream of;
    // std::ifstream in(filename, std::ios::in | std::ios::binary);
    //of.open("outfile.txt"); 
    int cols = 0, rows = 0;
    double buff[10000];

    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(filename);
    while (! infile.eof())
        {
        std::string line;
        std::getline(infile, line);

        int temp_cols = 0;
        std::stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
        }

    infile.close();

    rows--;
    //of << "reading params..." << std::endl;
    // Populate matrix with numbers.
    Eigen::MatrixXf result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            result(i,j) = buff[ cols*i+j ];
            //of << buff[ cols*i+j ] << std::endl;
        }

/*
    of << "rows/cols:" << rows << ' ' << cols << std::endl;
    of.close();*/
    return result;
};



void FixedUpdate( Eigen::Vector3f &state, Eigen::Vector2f control, float dt)
{
    state += dt * DDDynamics(state, control, CAR_LENGTH);
}

void testDynamics()
{
    Eigen::Vector3f x( 0.0f, 0.0f, 0.0f );
    Eigen::Vector2f u( 1.0f, -0.58f );

    std::cout << "Yo" << std::endl;
    
    std::cout << "initial: " << x << std::endl;
    float dt = DT;
    for (int i = 0; i < 10; ++i)
    {
        FixedUpdate( x, u, dt);
        std::cout << "state at time " << i * dt << std::endl;
        std::cout << x << std::endl << std::endl;
    }
}


void testEigen()
{
    std::vector<float> v(9);
    {
        int i = 0;
        for (float & f : v)
        {   
            f = i++;
            std::cout << f << std::endl;
        }
    }

    for (float f : v)
    {
        std::cout << f << std::endl;
    }

    Eigen::Matrix<float,3,3,Eigen::RowMajor> M;
    M << 0,1,2,3,4,5,6,7,8;
    std::cout << M << std::endl;
    for (int i = 0; i < 9; ++i)
    {   
        //dstd::cout << i << std::endl;
        // M(i,i) = i;
        std::cout << M(i) << std::endl;
    }

    Eigen::MatrixXf N((Eigen::MatrixXf)M);
    std::cout << N << std::endl;
    for (int i = 0; i < 9; ++i)
    {   
        //dstd::cout << i << std::endl;
        // M(i,i) = i;
        std::cout << N(i) << std::endl;
    }

}




class Network
{
public:
    Eigen::MatrixXf W1;
    Eigen::VectorXf B1;
    Eigen::MatrixXf W2;
    Eigen::VectorXf B2;

    Network()
    {
        std::cout << "Constructing network..." << std::endl;
        W1 = Eigen::MatrixXf(HIDDEN_SIZE,IN_SIZE);
        B1 = Eigen::VectorXf(HIDDEN_SIZE);
        W2 = Eigen::MatrixXf(OUT_SIZE,HIDDEN_SIZE);
        B2 = Eigen::VectorXf(OUT_SIZE);
        std::cout << "Network constructed..." << std::endl;
    }
    
    Eigen::Vector4f forward( Eigen::Vector3f state )
    {
        //std::cout << W1 * state << std::endl;
        //return Eigen::Vector2f::Zero();
        return W2 * (W1 * state + B1) + B2;
    }
};


float ReLU(float x)
{
    //std::normal_distribution<float> distribution(mean,stddev);
    
    return x > 0 ? x : 0;
}


Eigen::Vector4f Policy( Eigen::VectorXf params, Eigen::VectorXf feature )
{   
    // Eigen::Map<Eigen::MatrixXf>(zlin.data(),4,2)
    Eigen::MatrixXf W1 = Eigen::Map<Eigen::MatrixXf>(   params.head<IN_SIZE * HIDDEN_SIZE>().data(),
                                                        HIDDEN_SIZE,IN_SIZE);
    Eigen::VectorXf B1 = params.segment<HIDDEN_SIZE>(IN_SIZE * HIDDEN_SIZE);
    Eigen::MatrixXf W2 = Eigen::Map<Eigen::MatrixXf>(   params.segment<HIDDEN_SIZE * OUT_SIZE>(IN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE).data(),
                                                        OUT_SIZE, HIDDEN_SIZE);
    Eigen::VectorXf B2 = params.tail<OUT_SIZE>();

    //Eigen::MatrixXf W2;
    //Eigen::VectorXf B2;
    ////std::cout << W1 * state << std::endl;
    ////return Eigen::Vector2f::Zero();

    //std::cout << W2(0) << ' ' <<  W2(1) << ' ' << W2(2) << std::endl;
    return W2 * (W1 * feature + B1).unaryExpr(&ReLU) + B2;
    //return Eigen::Vector2f::Zero();
}

//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator (8980);
std::normal_distribution<float> distribution(0,1);
std::uniform_real_distribution<float> uniform(-1,1);
std::uniform_real_distribution<float> uniform01(0,1);


Eigen::Vector2f GetControl( Eigen::Vector2f mean, Eigen::Vector2f logstddev, float safety = 1.0f )
{
    Eigen::Vector2f u_t;

    if (USE_STOCHASTIC_SAMPLING)
    {
        u_t[0] = mean[0] + distribution(generator) * (logstddev[0] < -10000 ? 0 : expf( std::min( logstddev[0], safety )));
        u_t[1] = mean[1] + distribution(generator) * (logstddev[1] < -10000 ? 0 : expf( std::min( logstddev[1], safety )));
    }
    else
    {
        u_t = mean; // TODO WARNING GET RID OF THIS FOR STOCHASTIC TRAJECTORIES
    }
    if (CLIP_CONTROL)
    {
        u_t[0] = u_t[0] > 40 ? 40 : u_t[0];
        u_t[0] = u_t[0] < -40 ? -40 : u_t[0];
            
        u_t[1] = u_t[1] > 3.14159 ? 3.14159 : u_t[1];
        u_t[1] = u_t[1] < -3.14159 ? -3.14159 : u_t[1];
    }

    return u_t;
}


float EvaluatePolicy(Eigen::VectorXf params, int n_steps, int n_evals, float dt)
{
    float reward = 0.0f;
    for (int e = 0; e < n_evals; ++e)
    {
        float temp_reward = 0;
        Eigen::Vector3f x_t = Eigen::Vector3f(uniform01(generator) * -SIM_EXTENT, uniform(generator) * SIM_EXTENT, uniform(generator) * 3.14159); //Eigen::Vector3f::Zero();//Eigen::Vector3f(uniform01(generator) * -SIM_EXTENT, uniform(generator) * SIM_EXTENT, uniform(generator) * 3.14159); //Eigen::Vector3f::Zero();
        for (int n = 0; n < n_steps; ++n)
        {
            for (int i = 0; i < params.size(); ++i)
            {
                if( isnan(params[i]) )
                    std::cout << "NAN" << std::endl;
                if( isinf(params[i]) )
                    std::cout << "INF" << std::endl;
            }
            
            
            Eigen::VectorXf feature = GenFeature(x_t,GOAL_STATE);
            Eigen::Vector4f result = Policy(params,feature);
            for (int i = 0; i < result.size(); ++i)
            {
                if( isnan(result[i]) )
                    std::cout << "NAN" << std::endl;
                // if( isinf(result[i]) )
                //     std::cout << "INF" << std::endl;
            }
            Eigen::Vector2f mean = result.head<2>();
            Eigen::Vector2f logstddev = result.tail<2>();

            Eigen::Vector2f u_t;
            // u_t[0] = mean[0] + distribution(generator) * expf( std::min( logstddev[0], 10.0f ));
            // u_t[1] = mean[1] + distribution(generator) * expf( std::min( logstddev[1], 10.0f ));
            
            u_t = GetControl(mean,logstddev);

            //if (USE_STOCHASTIC_SAMPLING)
            //{
            //    u_t = GetControl(mean,logstddev);
            //}
            //else
            //{
            //    u_t = mean; // TODO WARNING GET RID OF THIS FOR STOCHASTIC TRAJECTORIES
            //}
            //
            //if (CLIP_CONTROL)
            //{
            //    u_t[0] = u_t[0] > 40 ? 40 : u_t[0];
            //    u_t[0] = u_t[0] < -40 ? -40 : u_t[0];
            //
            //    u_t[1] = u_t[1] > 3.14159 ? 3.14159 : u_t[1];
            //    u_t[1] = u_t[1] < -3.14159 ? -3.14159 : u_t[1];
            //}


            for (int i = 0; i < u_t.size(); ++i)
            {
                if( isnan(u_t[i]) )
                    std::cout << "NAN" << std::endl;
                if( isinf(u_t[i]) )
                    std::cout << "INF" << std::endl;
            }

            FixedUpdate(x_t, u_t, dt);
            for (int i = 0; i < x_t.size(); ++i)
            {
                if( isnan(x_t[i]) )
                    std::cout << "NAN" << std::endl;
                if( isinf(x_t[i]) )
                    std::cout << "INF" << std::endl;
            }
            /*Eigen::Vector3f displacement = x_t - GOAL_STATE;
            float dist = displacement.norm();
            float temp = 1.0/(dist+.1f) - u_t.squaredNorm()*dt*.01f;*/
            temp_reward += pow(.95,n) * GetReward(x_t,u_t,GOAL_STATE,dt);
        }
        // std::cout << "\t i.rew: " << temp_reward << std::endl;
        reward += temp_reward;
    }
    return reward/n_evals;
}


//float Reward(Eigen::VectorXf params, Eigen::Vector3f state)
//{
//    return 1.0f;
//}

// sorts in decreasing order for CEM
// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T>
std::vector<unsigned int> sort_indexes(const std::vector<T> &v)
{
  // initialize original index locations
  std::vector<unsigned int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](unsigned int i1, unsigned int i2) {return v[i1] > v[i2];});

  return idx;
}


float AddNoise(float mean, float stddev)
{
    //std::normal_distribution<float> distribution(mean,stddev);
    float rand = distribution(generator);
    //std::cout << mean << std::endl;
    if ( isnan(rand) || isnan(mean) || isnan(stddev) )
        std::cout << "OOPS" << std::endl;
    return mean + rand * stddev;
}

Eigen::VectorXf CEM(int blank, Eigen::VectorXf & theta_mean, int batch_size, int n_iter, float elite_frac, float initial_std=0.1f)
{
    #ifdef SAVE_OUTPUT
        std::ofstream outfile;
        outfile.open("new_results.csv");
    #endif
    int n_elite = int(roundf(elite_frac * batch_size));
    Eigen::VectorXf theta_std = Eigen::VectorXf::Ones(theta_mean.size()) * initial_std; // TODO: Make static

    for( int i = 0; i < n_iter; ++i )
    {
        std::vector<Eigen::VectorXf> thetas(batch_size);
        std::vector<float> rewards(batch_size);
        //std::vector<unsigned int> elite(batch_size);
        //std::cout << "thetas: " << thetas.size() << std::endl;
        //
        if (RUN_PARALLEL)
        {
            #pragma omp parallel for
            for (int k = 0; k < thetas.size(); ++k)
            {
                thetas.at(k) = theta_mean.binaryExpr(theta_std, &AddNoise);
                rewards.at(k) = EvaluatePolicy(thetas.at(k),TRAJECTORY_SAMPLES,EVALUATION_SAMPLES,DT); // Reward(theta,Eigen::Vector3f::Zero());
            
                //++k;
            }
        } 
        else
        {
            //int k = 0;
            for (int k = 0; k < thetas.size(); ++k)
            {
                thetas.at(k) = theta_mean.binaryExpr(theta_std, &AddNoise);
                rewards.at(k) = EvaluatePolicy(thetas.at(k),TRAJECTORY_SAMPLES,EVALUATION_SAMPLES,DT); // Reward(theta,Eigen::Vector3f::Zero());
                //++k;
            }
        }

        //for (Eigen::VectorXf & theta : thetas)
        //{
        //    theta = theta_mean.binaryExpr(theta_std, &AddNoise);
        //    rewards.at(k) = EvaluatePolicy(theta,TRAJECTORY_SAMPLES,30,DT); // Reward(theta,Eigen::Vector3f::Zero());
        //    
        //    ++k;
        //}

        std::vector<unsigned int> elite = sort_indexes(rewards);
        std::cout << i+1 << "/" << n_iter << ", Best reward: " << rewards[elite[0]] << std::endl;
        #ifdef SAVE_OUTPUT
            outfile << rewards[elite[0]] << std::endl;
        #endif
        Eigen::VectorXf new_mean = Eigen::VectorXf::Zero(theta_mean.size());
        Eigen::VectorXf new_std = Eigen::VectorXf::Zero(theta_mean.size());
        for(int e = 0; e < n_elite; ++e)
        {
            new_mean += thetas[elite[e]];
        }
        new_mean *= 1.0f/n_elite;


        // Assuming statistical independence between each variable
        for(int j = 0; j < new_mean.size(); ++j)
        {
            for(int e = 0; e < n_elite; ++e)
            {
                new_std[j] += powf((new_mean[j]-thetas[elite[e]][j]),2.0);
            }
            new_std[j] /= (n_elite-1); // TODO: maybe add a little bit to avoid vanishing std_dev (.1/(i+1))
            if (CEM_ADD_NOISE)
            {
                new_std[j] += (.1/(i*CEM_NOISE_FACTOR+1));
            }
        }


        // Set up new distribution
        theta_mean = new_mean;
        theta_std = new_std;
        
        

        /*for(int e = 0; e < n_elite; ++e)
        {
            Eigen::VectorXf displacement = thetas[elite[e]] - new_mean;
            new_std += displacement.squaredNorm();
        }*/
    }
    #ifdef SAVE_OUTPUT
        outfile.close();
    #endif
    //std::cout << th_std.size() << std::endl;
    ////Add noise to batch_size samples 
    //ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
    ////Evaluate each sample
    //ys = np.array([f(th,evalation_samples) for th in ths])
    //// Keep top n_elite best samples
    //elite_inds = ys.argsort()[::-1][:n_elite]
    //elite_ths = ths[elite_inds]
    //// Compute the mean and std-dev of best samples
    //th_mean = np.median(elite_ths,axis=0)
    //th_std = elite_ths.std(axis=0)
    ////  some extra noise
    //th_std += cem_noise_factor/(iter+1)
    ////Return results 
    return theta_mean;
}


float testfn(float x, float y)
{
    std::cout << "IN" << std::endl;
    std::cout << x << ',' << y << std::endl;
    return x + y;
}

// TODO: REMOVE THIS FOR PERFORMANCE
//unsigned int fp_control_state = _controlfp(_EM_ZERODIVIDE | _EM_INVALID, _MCW_EM);


int main()
{
    /*Eigen::VectorXf v_ran  = Eigen::VectorXf::Random(28);
    SaveFile("random.m",v_ran);
    Eigen::VectorXf rv_ran = ReadFile("random.m");
    std::cout << v_ran << std::endl << std::endl;
    std::cout << rv_ran << std::endl << "test" << std::endl << rv_ran.size() << std::endl;
    return 0;*/


    Eigen::Vector3f x( -25.0f, 0.0f, 0.75f );//x( 0.0f, 0.0f, 0.0f );
    Eigen::Vector2f u( 1.0f, -0.58f );
    Eigen::MatrixXf X(4,2), Y(4,2), Z(4,2);
    X << 1,2,3,4,5,6,7,8;
    Y << 1,1,1,1,1,1,1,1;

    std::cout << "Let's begin!" << std::endl;
    std::cout << 'x' << x.head<3>() << std::endl;
    //std::cout << X << std::endl;
    //std::cout << Y << std::endl;

    Z = Y.binaryExpr(X,&testfn);
    std::cout << Z << std::endl;
    Eigen::Map<Eigen::VectorXf> zlin(Z.data(),Z.size());
    // std::cout << Eigen::Map<Eigen::MatrixXf>(Z.data(),2,4) << std::endl;
    std::cout << zlin << std::endl;
    std::cout << Eigen::Map<Eigen::MatrixXf>(zlin.data(),4,2) << std::endl;



    Eigen::VectorXf params = Eigen::VectorXf::Zero(HIDDEN_SIZE * IN_SIZE + HIDDEN_SIZE* OUT_SIZE + OUT_SIZE + HIDDEN_SIZE);
    //params = params * 0.0f;
    //params[24] = 1;
    //params[25] = -.58;
    //params[26] = -1000;
    //params[27] = -1000;
    //params.tail<OUT_SIZE>() = Eigen::Vector4f(1,-.58,-1000,-1000);
    /*for (int i = 0; i < params.size(); ++i)
    {
        if( isnan(params[i]) )
            std::cout << "NAN" << std::endl;
        if( isinf(params[i]) )
            std::cout << "INF" << std::endl;
    }*/
    
    Eigen::VectorXf feature = GenFeature(x,GOAL_STATE);
    Eigen::VectorXf p = Policy(params,feature);
    std::cout << "Before: " << p << std::endl;
    std::cout << "Before: " << GetControl(p.head<2>(), p.tail<2>());


    //// SaveFile("test.m",params);
    //Eigen::write_binary("test.m",params);
    //Eigen::VectorXf vec;
    //Eigen::read_binary("test.m",vec);
    //std::cout << "Params: " << params << std::endl;
    //std::cout << "vec: " << vec << std::endl;
    //std::cout << (vec == params) << std::endl;

    //return 0;
    if (LOAD_PARAMETERS)
    {
        params = ReadFile("dd_parameters_3_200_4.m"); //new_parameters
    }
    else
    {
        std::cout << "Running CEM..." << std::endl;
        CEM(0,params,CEM_BATCH_SIZE,CEM_ITERATIONS,CEM_ELITE_FRAC);
        #ifdef SAVE_OUTPUT
                std::cout << "Saving parameters to disk..." << std::endl;
                //Eigen::write_binary("parameters2.m",params);
                SaveFile("new_parameters.m",params);
                std::cout << "Saved!" << std::endl;
        #endif
    }
    //return 0;

    
    /*Eigen::VectorXf*/ p = Policy(params,feature);
    
    std::cout << "After: " << p << std::endl;
    std::cout << "After: " << GetControl(p.head<2>(), p.tail<2>()) << std::endl;
    std::cout << "**************************************\nTEST\n**************************************" << std::endl;
    Eigen::Vector3f x_t = x;
    for (int n = 0; n < TRAJECTORY_SAMPLES; ++n)
    {
            Eigen::VectorXf feature = GenFeature(x_t,GOAL_STATE);
            Eigen::Vector4f result = Policy(params,feature);
            Eigen::Vector2f mean = result.head<2>();
            Eigen::Vector2f logstddev = result.tail<2>();

            Eigen::Vector2f u_t = GetControl(mean,logstddev);

            FixedUpdate(x_t, u_t, DT);
            Eigen::Vector3f displacement = x_t - GOAL_STATE;
            float dist = displacement.head<2>().norm();
            std::cout << "dist: " << dist << "\nx: " << x_t << "\nu: " << u_t << std::endl;
    }
    
    std::cout << "\n**********************************\nRandomTests\n***********************************\n" << std::endl;

    int counter = 0;
    int ntests = 100;
    for (int t = 0; t < ntests; ++t)
    {
        std::cout << std::endl << t << std::endl;
        Eigen::Vector3f x_t = Eigen::Vector3f(uniform(generator)*SIM_EXTENT, uniform(generator)*SIM_EXTENT, uniform(generator)*3.14159);
        std::cout << "Starting state:\n " << x_t << std::endl;
        for (int n = 0; n < TRAJECTORY_SAMPLES; ++n)
        {
                Eigen::VectorXf feature = GenFeature(x_t,GOAL_STATE);
                Eigen::Vector4f result = Policy(params,feature);
                Eigen::Vector2f mean = result.head<2>();
                Eigen::Vector2f logstddev = result.tail<2>();

                Eigen::Vector2f u_t = GetControl(mean,logstddev);

                FixedUpdate(x_t, u_t, DT);
                //Eigen::Vector3f displacement = x_t - GOAL_STATE;
                //float dist = displacement.head<2>().norm();
                //std::cout << "dist: " << dist << "\nx: " << x_t << std::endl;
        }
        Eigen::Vector2f displacement = x_t.head<2>() - GOAL_STATE.head<2>();
        float dist = displacement.norm();
        if (dist > 5) counter += 1;
        std::cout << "Final distance: " << dist << std::endl;
    }
    std::cout << std::endl << "Bad runs: " << counter << '/' << ntests << std::endl;

    //std::cout << "**************************************\nTRUTH\n**************************************" << std::endl;
    //x_t = x;
    //for (int n = 0; n < TRAJECTORY_SAMPLES; ++n)
    //{
    //        //Eigen::Vector4f result = Policy(params,x_t);
    //        //Eigen::Vector2f mean = result.head<2>();
    //        //Eigen::Vector2f logstddev = result.tail<2>();

    //        //Eigen::Vector2f u_t = GetControl(mean,logstddev);

    //        FixedUpdate(x_t, u, DT);
    //        Eigen::Vector3f displacement = x_t - GOAL_STATE;
    //        float dist = displacement.head<2>().norm();
    //        std::cout << "dist: " << dist << "\nx: " << x_t << std::endl;
    //}




    /*for(int i = 0; i < 10; ++i)
    {
        std::cout << AddNoise(0,.1) << std::endl;
    }*/

    /*Network n;
    std::cout << n.forward(x) << std::endl;*/
    //std::vector<float> v(12);
    //for(int i = 0; i < 12; ++i)
    //{
    //    v[i] = distribution(generator);
    //    std::cout<< v[i] << std::endl;
    //}
    //std::vector<unsigned int> result = sort_indexes(v);

    //
    //for(int i = 0; i < 12; ++i)
    //{
    //    std::cout<< result[i] << std::endl;
    //}

    return 0;
}