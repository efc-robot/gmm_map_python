import pygraph
from autolab_core import RigidTransform

class TFGraph(pygraph.DirectedGraph):
    def new_tf_node(self, robot_id):
        node_id = self.new_node()
        self.nodes[node_id]['data']['robot_id'] = robot_id

    def get_tf_node(self, robot_id):
        for key,node in self.nodes.items():
            if node['data']['robot_id'] == robot_id:
                return key,node
        # raise NonexistentNodeError(robot_id)
        return None,None

    def new_tf_edge(self, from_robot_id, to_robot_id, Rigidtrans):
        from_node_id,from_node = self.get_tf_node(from_robot_id)
        to_node_id,to_node = self.get_tf_node(to_robot_id)
        edge_id = self.new_edge(from_node_id, to_node_id, 1)
        self.edges[edge_id]['data']['tf'] = Rigidtrans
        edge_id = self.new_edge(to_node_id, from_node_id, 1)
        self.edges[edge_id]['data']['tf'] = Rigidtrans.inverse()
    
    def add_existing_tf_edge(self, from_robot_id, to_robot_id):
        edge_key,edge = self.get_tf_edge(from_robot_id, to_robot_id)
        edge['cost'] = 1.0/((1.0/edge['cost'])+1)
        edge_key,edge = self.get_tf_edge(to_robot_id, from_robot_id)
        edge['cost'] = 1.0/((1.0/edge['cost'])+1)

    def set_tf_edge(self, from_robot_id, to_robot_id, Rigidtrans):
        from_node_id,from_node = self.get_tf_node(from_robot_id)
        to_node_id,to_node = self.get_tf_node(to_robot_id)
        print 'change from'
        print self.get_graph_edge(from_node_id, to_node_id)[1]['data']['tf']
        print 'to'
        print Rigidtrans
        self.get_graph_edge(from_node_id, to_node_id)[1]['data']['tf'] = Rigidtrans
        self.get_graph_edge(to_node_id, from_node_id)[1]['data']['tf'] = Rigidtrans.inverse()

    def get_graph_edge(self, from_node_id, to_node_id):
        for edge_id in self.nodes[from_node_id]['edges']:
            if self.edges[edge_id]['vertices'][1] == to_node_id:
                return edge_id,self.edges[edge_id]
        return None,None
    
    def get_tf_edge(self, from_robot_id, to_robot_id):
        from_node_id,from_node = self.get_tf_node(from_robot_id)
        to_node_id,to_node = self.get_tf_node(to_robot_id)
        return self.get_graph_edge(from_node_id, to_node_id)

    def get_tf(self, from_robot_id, to_robot_id):
        from_node_id,from_node = self.get_tf_node(from_robot_id)
        to_node_id,to_node = self.get_tf_node(to_robot_id)
        path = pygraph.a_star_search(self, from_node_id, to_node_id)
        if path == []:
            return None
        elif len(path) == 1:
            return RigidTransform()
        result = self.get_graph_edge(path[0], path[1])[1]['data']['tf']
        for i in range(1,len(path)-1):
            result = self.get_graph_edge(path[i], path[i+1])[1]['data']['tf'] * result
        return result

    def get_connected_nodes(self, node_id):
        components = pygraph.get_connected_components(self)
        result_node_ids = []
        for node_list in components:
            if node_id in node_list:
                result_node_ids = node_list
        return result_node_ids

    def get_connected_robots(self, robot_id):
        node_id,node = self.get_tf_node(robot_id)
        result_node_ids = self.get_connected_nodes(node_id)
        result_robot_ids = []
        for con_node_id in result_node_ids:
            result_robot_ids.append(self.nodes[con_node_id]['data']['robot_id'])
        return result_robot_ids

    def get_tf_tree(self, robot_id):
        node_id,node = self.get_tf_node(robot_id)
        connected_node_ids = self.get_connected_nodes(node_id)
        connected_edge_ids = []
        for edge_id in self.get_all_edge_ids():
            if (self.edges[edge_id]['vertices'][0] in connected_node_ids) and (self.edges[edge_id]['vertices'][1] in connected_node_ids):
                connected_edge_ids.append(edge_id)
        subgraph = pygraph.make_subgraph(self, connected_node_ids, connected_edge_ids)
        mst = pygraph.find_minimum_spanning_tree(subgraph)
        tf_tree_list = []
        for tree_edge in mst:
            tf_tree_list.append(self.edges[tree_edge]['data']['tf'])
        return tf_tree_list

    def graph_to_dot(self, node_renderer=None, edge_renderer=None):
        """Produces a DOT specification string from the provided graph."""
        node_pairs = list(self.nodes.items())
        edge_pairs = list(self.edges.items())

        if node_renderer is None:
            node_renderer_wrapper = lambda nid: ''
        else:
            node_renderer_wrapper = lambda nid: ' [%s]' % ','.join(
                ['%s=%s' % tpl for tpl in list(node_renderer(self, nid).items())])

        # Start the graph
        graph_string = 'digraph G {\n'
        graph_string += 'overlap=scale;\n'

        # Print the nodes (placeholder)
        for node_id, node in node_pairs:
            graph_string += '%i : %i%s;\n' % (node_id, node['data']['robot_id'], node_renderer_wrapper(node_id))

        # Print the edges
        for edge_id, edge in edge_pairs:
            node_a = edge['vertices'][0]
            node_b = edge['vertices'][1]
            graph_string += '%i -> %i : %f : %s;\n' % (self.get_node(node_a)['data']['robot_id'], self.get_node(node_b)['data']['robot_id'], edge['cost'], str(edge['data']['tf']))

        # Finish the graph
        graph_string += '}'

        return graph_string

if __name__ == '__main__':
    tf_graph = TFGraph()
    tf_graph.new_tf_node(1)
    print tf_graph.graph_to_dot()
    tf_graph.new_tf_node(3)
    print tf_graph.graph_to_dot()
    tf_graph.new_tf_node(2)
    print tf_graph.graph_to_dot()
    tf_graph.new_tf_node(5)
    print tf_graph.graph_to_dot()
    tf = RigidTransform(from_frame = '1', to_frame = '2')
    tf_graph.new_tf_edge(1,2,tf)
    print tf_graph.graph_to_dot()
    tf = RigidTransform(from_frame = '2', to_frame = '3')
    tf_graph.new_tf_edge(2,3,tf)
    print tf_graph.graph_to_dot()
    tf = RigidTransform(from_frame = '3', to_frame = '5')
    tf_graph.new_tf_edge(3,5,tf)
    print tf_graph.graph_to_dot()
    tf = tf_graph.get_tf_tree(1)
    print tf
    import pdb; pdb.set_trace()