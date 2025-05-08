#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cmath>
#include <set>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include<climits>
#include<numeric>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>


#define NVR_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "Assertion failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            assert(condition); \
        } \
    } while(0)

enum NVR_Direction
{
  NVR_DIR_HORIZONTAL,
  NVR_DIR_VERTICAL,
  NVR_DIR_BOTH,
  NVR_DIR_NONE
};


class NVR_Point
{
  public:
    NVR_Point() {}
    NVR_Point(int x, int y) {m_x = x; m_y = y;}
    void set(int x, int y) {m_x = x; m_y = y;}
    int x() const { return m_x;}
    int y() const { return m_y;}
    void set_x(int x) { m_x = x;}
    void set_y(int y) { m_y = y;}

  private:
    int m_x, m_y;
};

class NVR_Box
{
  public:
    NVR_Box() : lo(INT_MAX, INT_MAX), hi(INT_MIN, INT_MIN) {}
    void update(int x, int y) {update_x(x); update_y(y);}
    void update_x(int x) {
      if(x < lo.x()) { lo.set_x(x);}
      if(x > hi.x()) { hi.set_x(x);}
    }
    void update_y(int y) {
      if(y < lo.y()) { lo.set_y(y);}
      if(y > hi.y()) { hi.set_y(y);}
    }

    int hpwl() const {return width() + height(); }
    int width() const { return hi.x() - lo.x(); }
    int height() const { return hi.y() - lo.y(); }
  private:
    NVR_Point lo, hi;
};


class NVR_Point3D
{
  public:
    NVR_Point3D(unsigned x, unsigned y, unsigned z) : m_x(x),
      m_y(y), m_z(z) {}

    unsigned x() const {return m_x;}
    unsigned y() const {return m_y;}
    unsigned z() const {return m_z;}
  private:
    unsigned m_x;
    unsigned m_y;
    unsigned m_z;
};

typedef NVR_Point3D NVR_Access;

class NVR_Pin
{
  public:
    unsigned num_accesses() const {return m_access.size();}
    void add_access(unsigned x, unsigned y, unsigned z) {
      m_access.emplace_back(x, y, z);
    }
    const std::vector<NVR_Access> &access() const {return m_access;}
  private:
    std::vector<NVR_Access> m_access;
};

class NVR_Net
{
  public:
    NVR_Net(const std::string &name) : m_name(name){}
    const std::string &name() const { return m_name;}
    void set_name(const std::string &name) {m_name = name;}
    NVR_Pin &add_pin() {m_pins.emplace_back(); return m_pins.back();}
    unsigned idx() const { return m_idx;}
    void set_idx(unsigned idx) { m_idx = idx; }
    const std::vector<NVR_Pin> &pins() const { return m_pins; }

    NVR_Box &box() { return const_cast<NVR_Box &>(const_cast<const NVR_Net *>(this)->box());}
    const NVR_Box &box() const { return m_box;}
  private:
    std::string m_name;
    unsigned m_idx;
    std::vector<NVR_Pin> m_pins;
    NVR_Box m_box;

};


class NVR_Gcell
{
  public:
    void incr_demand(unsigned demand) { m_demand += demand;}
    void decr_demand(unsigned demand) { m_demand -= demand;}
    void set_demand(unsigned demand) { m_demand = demand;}
    unsigned demand() const {return m_demand;}

    double capacity() const {return m_capacity;}
    void set_capacity(double cap) {m_capacity = cap;}
  private:
    unsigned m_demand : 16;
    double m_capacity;
};

class NVR_GridGraph2D
{
  public:
    NVR_GridGraph2D() : m_dir(NVR_DIR_NONE){};
    void set_name(const std::string &name) {m_name = name;}
    const std::string &name() const { return m_name;}

    void init(unsigned gridx, unsigned gridy);
    void set_direction(int dr);
    bool is_routing_layer() const { return m_dir != NVR_DIR_NONE;}
    bool is_hor() const {return m_dir == NVR_DIR_HORIZONTAL; }
    bool is_ver() const {return m_dir == NVR_DIR_VERTICAL;}

    void set_min_length(double min_length) {m_min_length = min_length;}
    double min_length() const {return m_min_length;}
    void set_unit_cost(double length, double via, double overflow);
    double unit_length_cost() const {return m_unit_length_cost;}
    double unit_via_cost() const {return m_unit_via_cost;}
    double unit_overflow_cost() const {return m_unit_overflow_cost;}

    NVR_Gcell &get_gcell(unsigned x, unsigned y);
  private:
    std::string m_name;
    NVR_Direction m_dir;
    double m_min_length;
    double m_unit_length_cost;
    double m_unit_via_cost;
    double m_unit_overflow_cost;

    unsigned m_num_gridx;
    unsigned m_num_gridy;
    std::vector<NVR_Gcell> m_gcells;
};

void NVR_GridGraph2D::set_unit_cost(double length, double via,
  double overflow) {
  m_unit_length_cost = length;
  m_unit_via_cost = via;
  m_unit_overflow_cost = overflow;
}

void NVR_GridGraph2D::set_direction(int dir)
{
  if(dir == 0) {
    m_dir = NVR_DIR_HORIZONTAL;
  } else if (dir == 1) {
    m_dir = NVR_DIR_VERTICAL;
  } else {
    fprintf(stderr, "Invalid direction value in set_direction: %d\n", dir);
    NVR_ASSERT(0);
  }
}

void NVR_GridGraph2D::init(unsigned gridx, unsigned gridy)
{
  m_num_gridx = gridx;
  m_num_gridy = gridy;
  m_gcells.resize(gridx * gridy);
}

NVR_Gcell &NVR_GridGraph2D::get_gcell(unsigned x, unsigned y)
{
  if (!(x < m_num_gridx && y < m_num_gridy && is_routing_layer())) {
    if (x >= m_num_gridx) {
      fprintf(stderr, "Grid x coordinate out of bounds: x=%u (max x=%u)\n", x, m_num_gridx);
    }
    if (y >= m_num_gridy) {
      fprintf(stderr, "Grid y coordinate out of bounds: y=%u (max y=%u)\n", y, m_num_gridy);
    }
    if (!is_routing_layer()) {
      fprintf(stderr, "Invalid layer type - attempting to access non-routing layer\n");
    }
    NVR_ASSERT(0);
  }
  if(is_hor()) {
    return m_gcells[y * m_num_gridx + x];
  } else {
    return m_gcells[x * m_num_gridy + y];
  }
}

class NVR_GridGraph
{
  public:
    unsigned num_gridx() const { return m_num_gridx; };
    unsigned num_gridy() const { return m_num_gridy; };
    unsigned num_layer() const { return m_num_layer; };
    void init(unsigned x, unsigned y, unsigned z);
    NVR_GridGraph2D &plane(unsigned layer) {return m_plane[layer];}
    const NVR_GridGraph2D &plane(unsigned layer) const {return m_plane[layer];}

    void init_x_coords(std::vector<int> &coord) { m_x_coords = coord; }
    void init_y_coords(std::vector<int> &coord) { m_y_coords = coord; }
    int cell_width(int x) const { return m_x_coords[x + 1] - m_x_coords[x];}
    int cell_height(int y) const { return m_y_coords[y + 1] - m_y_coords[y];}
  private:
    unsigned m_num_layer;
    unsigned m_num_gridx;
    unsigned m_num_gridy;
    std::vector<NVR_GridGraph2D> m_plane;
    std::vector<int> m_x_coords;
    std::vector<int> m_y_coords;
};


class NVR_DB
{
  public:

    bool read_files(int argc, char *argv[]);
    void profile();

  private:
    bool read_graph(const char *);
    bool read_nets(const char *);
    bool read_gr_solution(const char *);
    void report_statistic();
    bool check_connectivity(const NVR_Net *net,
      std::vector< std::vector< std::vector<int> > > &flag) const;
    void update_stacked_via_counter(unsigned net_idx, const std::vector<NVR_Point3D> &via_loc,
          std::vector< std::vector< std::vector<int> > > &flag,
          std::vector< std::vector< std::vector<int> > > &stacked_via_counter) const;

    double overflowLossFunc(double overflow, double slope);

    std::vector<NVR_Net> m_nets;
    NVR_GridGraph m_graph;
    std::vector<int> layer_directions;
};

void segv_handler(int sig) {
  void *array[1024];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

int main(int argc, char* argv[]) {

  signal(SIGSEGV, segv_handler);
  NVR_DB rdb;
  if( !rdb.read_files(argc, argv)) {
    printf("input error\n");
    return 1;
  }
  //rdb.profile();
  return 0;
}

bool NVR_DB::read_files(int argc, char* argv[])
{
  if (argc < 4) {
    printf("Usage %s resource_file net_file GR_file", argv[0]);
    return false;
  }

  if(!read_graph(argv[1])) {
    return false;
  }

  if(!read_nets(argv[2])) {
    return false;
  }

  report_statistic();

  if(!read_gr_solution(argv[3])) {
    return false;
  }

  return true;
}

void NVR_DB::report_statistic()
{
  printf("Num nets = %ld\n", m_nets.size());
  printf("Grid Graph Size (x, y, z)= %d x %d x %d\n",
        m_graph.num_gridx(), m_graph.num_gridy(), m_graph.num_layer());
  return;
}

bool NVR_DB::read_graph(const char *input)
{
  std::ifstream fin(input);
  if (!fin) {
    printf("Failed to open resource file.\n");
    return false;
  }

  unsigned num_gridx, num_gridy, num_layers;
  fin >> num_layers;
  fin >> num_gridx;
  fin >> num_gridy;
  m_graph.init(num_gridx, num_gridy, num_layers);

  //printf("tmp1  %d %d %d\n", num_gridx, num_gridy, num_layers);

  double unit_length_cost, unit_via_cost;
  std::vector<double> overflow_costs;
  fin >> unit_length_cost;
  fin >> unit_via_cost;
  overflow_costs.resize(num_layers);
  layer_directions.resize(num_layers);
  for(unsigned z = 0; z < num_layers; z++) {
    fin >> overflow_costs[z];
    NVR_GridGraph2D &plane = m_graph.plane(z);
    plane.set_unit_cost(unit_length_cost,
      unit_via_cost, overflow_costs[z]);

    // printf("z=%d  lengtjh_cost=%.2lf  via_cost=%.2lf  overflow=%.2lf\n",
      // unit_length_cost, unit_via_cost, overflow_costs[z]);
  }

  int gcell_length;
  std::vector<int> coords;
  coords.resize(num_gridx);
  coords[0] = 0;
  for(unsigned x = 0; x < num_gridx - 1; x++) {
    fin >> gcell_length;
    coords[x + 1] = coords[x] + gcell_length;
    // printf("1 length=%d\n", gcell_length);
  }
  m_graph.init_x_coords(coords);

  coords.resize(num_gridy, 0);
  coords[0] = 0;
  for(unsigned y = 0; y < num_gridy - 1; y++) {
    fin >> gcell_length;
    coords[y + 1] = coords[y] + gcell_length;
    //printf("2 length=%d\n", gcell_length);
  }
  m_graph.init_y_coords(coords);



  std::string line;
  std::string layer_name;
  int direction;
  double capacity;
  double min_length;
  std::getline(fin, line);
  for(unsigned z = 0; z < num_layers; z++ ) {
    std::getline(fin, line);
    std::istringstream info(line);
    info >> layer_name;
    info >> direction;
    info >> min_length;
    NVR_GridGraph2D &plane = m_graph.plane(z);
    layer_directions[z] = direction;

    plane.set_name(layer_name);
    plane.set_min_length(min_length);
    printf("layer %d name=%s  min_length=%.2lf dir=%d\n",
      z, layer_name.c_str(), min_length, direction);

    if(z != 0) {
      plane.set_direction(direction);
      if(plane.is_routing_layer()) {
        plane.init(num_gridx, num_gridy);
      }
    }

    for(unsigned y = 0; y < num_gridy; y++) {
      std::getline(fin, line);
      std::istringstream info(line);
      for(unsigned x = 0; x < num_gridx; x++) {
          info >> capacity;
          if(plane.is_routing_layer()) {
            plane.get_gcell(x, y).set_capacity(capacity);
          }
      }
    }
  }

  return true;
}

bool NVR_DB::read_nets(const char *input)
{
  std::ifstream net_file(input);
  if (!net_file) {
    printf("Failed to open net file.\n");
    return false;
  }

  m_nets.reserve(1000);
  std::string line;
  std::string redundant_chars = "(),[]";
  while (std::getline(net_file, line)) {
    if (line.find("(") == std::string::npos && line.find(")")
      == std::string::npos && line.length()>1) {  //start to read a net
        size_t found = line.find('\n');
        if (found != std::string::npos) {
          line.erase(found, 1);
        }
      m_nets.emplace_back(line);
    } else if (line.find('[') != std::string::npos) { //read pins
      NVR_Net &net = m_nets.back();
      net.set_idx(m_nets.size() - 1);
      line.erase(std::remove_if(line.begin(), line.end(), [&redundant_chars](char c) {
              return redundant_chars.find(c) != std::string::npos;
              }), line.end());
      std::istringstream ss(line);

      NVR_Pin &pin = net.add_pin();
      int x, y, z;
      while (ss >> z >> x >> y) {
        //printf("access (x, y, z) = (%d, %d %d)\n", x, y, z);
        pin.add_access(x, y, z);
        net.box().update(x, y);
      }
    }
  }

  /*
  for(const NVR_Net &net : m_nets) {
    printf("net %s\n", net.name().c_str());
    for(const NVR_Pin &pin : net.pins()) {
      for(const NVR_Access &access : pin.access()) {
        printf("(%d, %d, %d) ", access.x(), access.y(), access.z());
      }
      printf("\n");
    }
  }
  */
  return true;
}

bool NVR_DB::read_gr_solution(const char *input)
{
  std::ifstream fin(input);
  if (!fin) {
    printf("Failed to open solution file.\n");
    return false;
  }

  std::unordered_map<std::string, NVR_Net *> net_mapper;
  std::unordered_map<std::string, bool> net_completed;
  for(NVR_Net &net : m_nets) {
    net_mapper[net.name()] = &net;
    net_completed[net.name()] = false;
  }

  unsigned long total_opens = 0;
  std::vector<int> total_vias(m_graph.num_layer(), 0);
  std::vector< std::vector< std::vector<int> > > flag;
  std::vector< std::vector< std::vector<int> > > wire_counter;
  std::vector< std::vector< std::vector<int> > > stacked_via_counter;

  flag.resize(m_graph.num_layer());
  wire_counter.resize(m_graph.num_layer());
  stacked_via_counter.resize(m_graph.num_layer());
  for(unsigned z = 0; z < m_graph.num_layer(); z++) {
    flag[z].resize(m_graph.num_gridx());
    wire_counter[z].resize(m_graph.num_gridx());
    stacked_via_counter[z].resize(m_graph.num_gridx());
    for(unsigned x = 0; x < m_graph.num_gridx(); x++) {
      flag[z][x].resize(m_graph.num_gridy(), -1);
      wire_counter[z][x].resize(m_graph.num_gridy(), 0);
      stacked_via_counter[z][x].resize(m_graph.num_gridy(), 0);
    }
  }

  std::vector<NVR_Point3D> via_loc;
  bool has_connectivity_violation = false;
  NVR_Net *net = NULL;
  std::string line;
  while(std::getline(fin, line)) {
    //printf("read %s\n", line.c_str());
    if(!net) {
      net = net_mapper[line];
      has_connectivity_violation = false;
    } else if(line[0] == '(') {
    } else if(line[0] == ')') {
      update_stacked_via_counter(net->idx(), via_loc, flag, stacked_via_counter);
      if(has_connectivity_violation) {
        total_opens++;
      } else {
        if (!net) {
          fprintf(stderr, "Null net pointer in read_gr_solution for net: %s\n", line.c_str());
          NVR_ASSERT(0);
        }
        if(!check_connectivity(net, flag)) {
          total_opens++;
        } else {
          net_completed[net->name()] = true;
        }
      }
      net = NULL;
      via_loc.clear();
    } else {
      //printf("wire %s\n", line.c_str());
      std::istringstream ss(line);
      int xl, yl, zl, xh, yh, zh;
      ss >> xl >> yl >> zl >> xh >> yh >> zh;
      //printf("(%d, %d, %d) (%d, %d, %d)\n", xl, yl, zl, xh, yh, zh);
      if(zh != zl) { // via
        if(xh == xl && yh == yl) {
          for(unsigned z = zl; z < zh; z++) {
            total_vias[z]++;
            via_loc.emplace_back(xl, yl, z);
          }
          // flag[zh][xl][yl] = net->idx();
        } else {
          fprintf(stderr, "Invalid via coordinates - x and y must be equal for vias:\n");
          fprintf(stderr, "  Start point: (%d,%d,%d)\n", xl, yl, zl);
          fprintf(stderr, "  End point:   (%d,%d,%d)\n", xh, yh, zh);
          if (xh != xl) {
            fprintf(stderr, "  x coordinates differ: %d != %d\n", xl, xh);
          }
          if (yh != yl) {
            fprintf(stderr, "  y coordinates differ: %d != %d\n", yl, yh);
          }
          NVR_ASSERT(0);
          has_connectivity_violation = true;
        }
      } else { //wire
        NVR_GridGraph2D &plane = m_graph.plane(zl);
        if(plane.is_hor()) {
          if (yh != yl || xh <= xl) {
            fprintf(stderr, "Invalid horizontal wire coordinates:\n");
            fprintf(stderr, "  Start point: (%d,%d,%d)\n", xl, yl, zl);
            fprintf(stderr, "  End point:   (%d,%d,%d)\n", xh, yh, zh);
            if (yh != yl) {
              fprintf(stderr, "  y coordinates differ: %d != %d\n", yl, yh);
            }
            if (xh <= xl) {
              fprintf(stderr, "  x end coordinate (%d) must be greater than x start coordinate (%d)\n", xh, xl);
            }
            NVR_ASSERT(0);
            has_connectivity_violation = true;
          }
        } else if(plane.is_ver()) {
          if (xh != xl || yh <= yl) {
            fprintf(stderr, "Invalid vertical wire coordinates:\n");
            fprintf(stderr, "  Start point: (%d,%d,%d)\n", xl, yl, zl);
            fprintf(stderr, "  End point:   (%d,%d,%d)\n", xh, yh, zh);
            if (xh != xl) {
              fprintf(stderr, "  x coordinates differ: %d != %d\n", xl, xh);
            }
            if (yh <= yl) {
              fprintf(stderr, "  y end coordinate (%d) must be greater than y start coordinate (%d)\n", yh, yl);
            }
            NVR_ASSERT(0);
            has_connectivity_violation = true;
          } else { //unroutable layer
            fprintf(stderr, "Invalid layer type - attempting to route on non-routing layer: layer=%d\n", zl);
            NVR_ASSERT(0);
            has_connectivity_violation = true;
          }
        }
      }
    }
  }
  double wl_cost = 0;
  double via_cost = 0;
  double overflow_cost = 0;
  double overflow_slope = 0.5;

  for(unsigned z = 0; z < m_graph.num_layer(); z++) {
    NVR_GridGraph2D &gg = m_graph.plane(z);
    if(!gg.is_routing_layer()) {
      via_cost += double(total_vias[z]) * gg.unit_via_cost();
      continue;
    }

    unsigned long long total_wl = 0;
    double layer_overflows = 0;
    double overflow = 0;
    for(unsigned x = 0; x < m_graph.num_gridx(); x++) {
      for(unsigned y = 0; y < m_graph.num_gridy(); y++) {
        NVR_Gcell &cell = gg.get_gcell(x, y);
        int demand = 2 * wire_counter[z][x][y];  //+ stacked_via_counter[z][x][y];
        cell.set_demand(demand);

        if (cell.capacity() > 0.001 ){
          overflow = double(cell.demand()) - 2 * cell.capacity();
          layer_overflows += overflowLossFunc(overflow/2, overflow_slope);
        } else if (cell.capacity() >= 0 && cell.demand() > 0) {
          // layer_overflows +=  overflowLossFunc(1.5 * double(cell.demand()), overflow_slope);
          layer_overflows +=  overflowLossFunc(double(cell.demand())/2, overflow_slope);
        } else if (cell.capacity() < 0) {
          printf("Capacity error (%d, %d, %d)\n", x, y, z);
        }

        if(gg.is_hor()) {
          total_wl += wire_counter[z][x][y] * m_graph.cell_width(x);
        } else if(gg.is_ver()) {
          total_wl += wire_counter[z][x][y] * m_graph.cell_height(y);
        }
      }
    }
    // overflow_cost += double(num_overflows) * 0.5 * gg.unit_overflow_cost();
    overflow_cost += layer_overflows * gg.unit_overflow_cost();
    via_cost += double(total_vias[z]) * gg.unit_via_cost();
    wl_cost += double(total_wl) * gg.unit_length_cost();
    printf("Layer = %d, layer_overflows = %lf, overflow cost = %lf\n", z, layer_overflows, overflow_cost);
  }

  unsigned long total_incompleted = 0;
  for (auto & [key, value] : net_completed) {
    if (value == false) {
      total_incompleted++;
    }
  }

  double total_cost = overflow_cost + via_cost + wl_cost;
  printf("Number of open nets : %lu\n", total_opens);
  printf("Number of incompleted nets : %lu\n", total_incompleted);
  printf("wirelength cost %.4lf\n", wl_cost);
  printf("via cost %.4lf\n", via_cost);
  printf("overflow cost %.4lf\n", overflow_cost);
  printf("total cost %.4lf\n", total_cost);
  return true;
}

void NVR_DB::update_stacked_via_counter(unsigned net_idx,
  const std::vector<NVR_Point3D> &via_loc,
  std::vector< std::vector< std::vector<int> > > &flag,
  std::vector< std::vector< std::vector<int> > > &stacked_via_counter) const
{
  for(const NVR_Point3D &pp : via_loc) {
    if(flag[pp.z()][pp.x()][pp.y()] != net_idx) {
      flag[pp.z()][pp.x()][pp.y()] = net_idx;

      int direction = layer_directions[pp.z()];
      if(direction == 0) {
        if ((pp.x() > 0) && (pp.x() < m_graph.num_gridx() - 1)) {
          stacked_via_counter[pp.z()][pp.x()-1][pp.y()]++;
          stacked_via_counter[pp.z()][pp.x()][pp.y()]++;
        } else if (pp.x() > 0 ) {
          stacked_via_counter[pp.z()][pp.x()-1][pp.y()] += 2;
        } else if (pp.x() < m_graph.num_gridx() - 1) {
          stacked_via_counter[pp.z()][pp.x()][pp.y()] += 2;
        }
      } else if (direction == 1) {
        if ((pp.y() > 0) && (pp.y() < m_graph.num_gridy() - 1)) {
          stacked_via_counter[pp.z()][pp.x()][pp.y()-1]++;
          stacked_via_counter[pp.z()][pp.x()][pp.y()]++;
        } else if (pp.y() > 0 ) {
          stacked_via_counter[pp.z()][pp.x()][pp.y()-1] += 2;
        } else if (pp.y() < m_graph.num_gridy() - 1) {
          stacked_via_counter[pp.z()][pp.x()][pp.y()] += 2;
        }
      }
      //stacked_via_counter[pp.z()][pp.x()][pp.y()]++;
    }

  }

  for(const NVR_Point3D &pp : via_loc) {
    flag[pp.z()][pp.x()][pp.y()] = net_idx;
    flag[pp.z()+1][pp.x()][pp.y()] = net_idx;
  }
}

bool NVR_DB::check_connectivity(const NVR_Net *net,
  std::vector< std::vector< std::vector<int> > > &flag) const
{
  int mark = net->idx();
  int traced_mark = net->idx() + m_nets.size();
  if (!net->pins().size()) {
    fprintf(stderr, "Net %s must have at least one pin for connectivity check\n", net->name().c_str());
    NVR_ASSERT(0);
  }
  //printf("net pins %d\n", net->pins().size());
  std::vector<NVR_Point3D> stack;
  for( const NVR_Access &ac : net->pins()[0].access()) {
    //printf("access %d %d %d\n", ac.x(), ac.y(), ac.z());
    if(flag[ac.z()][ac.x()][ac.y()] == mark) {
      flag[ac.z()][ac.x()][ac.y()] = traced_mark;
      stack.emplace_back(ac);
    }
  }
  while(!stack.empty()) {
    NVR_Point3D pp = stack.back();
    stack.pop_back();
    //printf("(%d %d %d)\n", pp.x(), pp.y(), pp.z());
    const NVR_GridGraph2D &gg = m_graph.plane(pp.z());
    if(gg.is_hor()) {
      //printf("west\n");
      if(pp.x() > 0 && flag[pp.z()][pp.x() - 1][pp.y()] == mark) {
        flag[pp.z()][pp.x() - 1][pp.y()] = traced_mark;
        stack.emplace_back(pp.x() - 1, pp.y(), pp.z());
      }
      //printf("east\n");
      if(pp.x() < m_graph.num_gridx() - 1 && flag[pp.z()][pp.x() + 1][pp.y()] == mark) { //west
        flag[pp.z()][pp.x() + 1][pp.y()] = traced_mark;
        stack.emplace_back(pp.x() + 1, pp.y(), pp.z());
      }
    } else if(gg.is_ver()) {
      //printf("south\n");
      if(pp.y() > 0 && flag[pp.z()][pp.x()][pp.y() - 1] == mark) {
        flag[pp.z()][pp.x()][pp.y() - 1] = traced_mark;
        stack.emplace_back(pp.x(), pp.y() - 1, pp.z());
      }
      //printf("north\n");
      if(pp.y() < m_graph.num_gridy() - 1 && flag[pp.z()][pp.x()][pp.y() + 1] == mark) { //west
        flag[pp.z()][pp.x()][pp.y() + 1] = traced_mark;
        stack.emplace_back(pp.x(), pp.y() + 1, pp.z());
      }
    }

    //printf("down\n");
    if(pp.z() > 0 && flag[pp.z() - 1][pp.x()][pp.y()] == mark) {
      flag[pp.z() - 1][pp.x()][pp.y()] = traced_mark;
      stack.emplace_back(pp.x(), pp.y(), pp.z() - 1);
    }

    //printf("up\n");
    if(pp.z() < m_graph.num_layer() - 1 && flag[pp.z() + 1][pp.x()][pp.y()] == mark) { //west
      flag[pp.z() + 1][pp.x()][pp.y()] = traced_mark;
      stack.emplace_back(pp.x(), pp.y(), pp.z() + 1);
    }
  }

  //printf("end propagate\n");
  for(unsigned i = 1; i < net->pins().size(); i++) {
    bool connected = false;
    for( const NVR_Access &ac : net->pins()[i].access()) {
      if(flag[ac.z()][ac.x()][ac.y()] == traced_mark) {
        connected = true;
        break;
      }
    }
    if(!connected) {
      return false;
    }
  }

  return true;
}

void NVR_DB::profile()
{
  if (!m_nets.size()) {
    fprintf(stderr, "No nets available for profiling - nets.size()=%zu\n", m_nets.size());
    NVR_ASSERT(0);
  }
  int max_hpwl = 0;
  double avg_hpwl = 0;
  std::vector<int> sorted_hpwl;
  sorted_hpwl.reserve(m_nets.size());
  for(const NVR_Net &net : m_nets) {
    int hpwl = net.box().hpwl();
    max_hpwl = std::max(max_hpwl, hpwl);
    avg_hpwl += hpwl;
    sorted_hpwl.emplace_back(hpwl);
  }
  avg_hpwl = avg_hpwl / double(m_nets.size());

  int top_10percent = m_nets.size() / 10;
  std::sort(sorted_hpwl.begin(), sorted_hpwl.end());
  double sum_10percent = std::accumulate(sorted_hpwl.begin(), sorted_hpwl.end(), 0);
  double avg_top_10percent = sum_10percent / double(top_10percent);


  printf("max_hpwl=%d  avg_hpwl=%.2lf  avg_hpwl[10%%]=%.2lf\n",
    max_hpwl, avg_hpwl, avg_top_10percent);
  return;
}


void NVR_GridGraph::init(unsigned x, unsigned y, unsigned z)
{
  m_num_gridx = x;
  m_num_gridy = y;
  m_num_layer = z;
  m_plane.resize(z);
}

double NVR_DB::overflowLossFunc(double overflow, double slope)
{
        return exp(overflow * slope);
}

