#include <cassert>
#include <iostream>

#include <boost/foreach.hpp>

#include <Array/Array.h>
#include <ReClaM/FFNet.h>
#include <ReClaM/createConnectionMatrix.h>

//From http://www.richelbilderbeek.nl/CppGetRandomUniform.htm
double GetRandomUniform()
{
  return static_cast<double>(std::rand())/static_cast<double>(RAND_MAX);
}

///NeuralNet is a derived class of FFNet
///to gain access to some protected methods of FFNet
struct NeuralNet : public FFNet
{
  NeuralNet(
    const int n_inputs,
    const int n_outputs,
    const Array<int> connection_matrix)
  : FFNet( n_inputs,n_outputs,connection_matrix) {}
  NeuralNet(const NeuralNet& n)
  : FFNet(n) {}

  void Activate(const Array<double> &in)
  {
    this->activate(in);
  }
  unsigned int GetNumberOfNeurons()
  {
    return this->numberOfNeurons;
  }
  void mutate(const double m)
  {
    Array<double> weights = this->getWeights();
    BOOST_FOREACH(double& x,weights)
    {
      x+= (GetRandomUniform() * (2.0 * m)) - m;
    }
    this->weightMatrix = weights;
  }
};

NeuralNet CreateNet(
  const int n_inputs,
  const int n_hidden,
  const int n_outputs,
  const double init_weight_min,
  const double init_weight_max)
{
  //Create neural net connection matrix
  Array<int> connection_matrix;
  createConnectionMatrix(connection_matrix,n_inputs, n_hidden, n_outputs);
  //Create the feed-forward neural network
  NeuralNet n(n_inputs, n_outputs, connection_matrix);
  n.initWeights(init_weight_min,init_weight_max);
  return n;
}

double Rate_xor_success(NeuralNet& n)
{
  double rating = 4.0;
  const unsigned int output_neuron_index = n.GetNumberOfNeurons() - 1;
  {
    std::vector<double> v(2);
    v[0] = 0.0;
    v[1] = 0.0;
    Array<double> inputs(v);
    n.Activate(inputs);
    const double output = n.outputValue(output_neuron_index);
    rating -= std::fabs(0.0 - output);
  }
  {
    std::vector<double> v(2);
    v[0] = 1.0;
    v[1] = 0.0;
    Array<double> inputs(v);
    n.Activate(inputs);
    const double output = n.outputValue(output_neuron_index);
    rating -= std::fabs(1.0 - output);
  }
  {
    std::vector<double> v(2);
    v[0] = 0.0;
    v[1] = 1.0;
    Array<double> inputs(v);
    n.Activate(inputs);
    const double output = n.outputValue(output_neuron_index);
    rating -= std::fabs(1.0 - output);
  }
  {
    std::vector<double> v(2);
    v[0] = 1.0;
    v[1] = 1.0;
    Array<double> inputs(v);
    n.Activate(inputs);
    const double output = n.outputValue(output_neuron_index);
    rating -= std::fabs(0.0 - output);
  }
  return rating;
}

int main()
{
  NeuralNet best_net = CreateNet(2,2,1,-1.0,1.1);
  double best_result = Rate_xor_success(best_net);

  for (int t=0; t!=1000000; ++t)
  {
    NeuralNet copy(best_net);
    copy.mutate(10.0);
    double result = Rate_xor_success(copy);
    if (result > best_result)
    {
      best_net = copy;
      best_result = result;
      std::cout << "Better result (t=" << t << "): "
        << Rate_xor_success(best_net) << std::endl;
    }
  }
}
