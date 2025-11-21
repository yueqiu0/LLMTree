from graphviz import Source

dot_code = '''
digraph Forest {
  rankdir=TB;
  labelloc="t";

  // IO-Tree
  subgraph cluster_IO {
    label="(a) IO-Tree";
    labelloc="b";
    labeljust="c";
    style=invis;

    node1 [label="Glucose < 125?"];
    node2 [label="BMI < 35?"];
    node3 [label="yes"];
    node4 [label="no"];
    node5 [label="Age < 35?"];
    node6 [label="yes"];
    node7 [label="no"];
    node_io_pad1 [label="", style=invis, width=0, height=0];

    node1 -> node2 [label="no"];
    node2 -> node3 [label="yes"];
    node2 -> node4 [label="no"];
    node1 -> node5 [label="yes"];
    node5 -> node6 [label="yes"];
    node5 -> node7 [label="no"];
    node4 -> node_io_pad1 [style=invis];  // 拉一层
  }

  // CoT-Tree
  subgraph cluster_CoT {
    label="(b) CoT-Tree";
    labelloc="b";
    labeljust="c";
    style=invis;

    cot_node1 [label="Glucose < 120?"];
    cot_node2 [label="BMI < 30?"];
    cot_node3 [label="yes"];
    cot_node4 [label="no"];
    cot_node5 [label="no"];
    cot_pad [label="", style=invis, width=0, height=0];

    cot_node1 -> cot_node2 [label="no"];
    cot_node2 -> cot_node3 [label="yes"];
    cot_node2 -> cot_node4 [label="no"];
    cot_node1 -> cot_node5 [label="yes"];
    cot_node4 -> cot_pad [style=invis];
  }

  // ToT-Tree
  subgraph cluster_ToT {
    label="(c) ToT-Tree";
    labelloc="b";
    labeljust="c";
    style=invis;

    tot_node1 [label="Blood Pressure < 74?"];
    tot_node2 [label="Glucose < 95?"];
    tot_node3 [label="yes"];
    tot_node4 [label="no"];
    tot_node5 [label="Glucose < 125?"];
    tot_node6 [label="yes"];
    tot_node7 [label="no"];
    tot_pad [label="", style=invis, width=0, height=0];

    tot_node1 -> tot_node2 [label="yes"];
    tot_node2 -> tot_node3 [label="yes"];
    tot_node2 -> tot_node4 [label="no"];
    tot_node1 -> tot_node5 [label="no"];
    tot_node5 -> tot_node6 [label="yes"];
    tot_node5 -> tot_node7 [label="no"];
    tot_node7 -> tot_pad [style=invis];
  }

  // LLMT-Tree
  subgraph cluster_LLMT {
    label="(d) LLMT-Tree";
    labelloc="b";
    labeljust="c";
    style=invis;

    llmt_node1 [label="Glucose < 125?"];
    llmt_node2 [label="no"];
    llmt_node3 [label="yes"];
    llmt_pad [label="", style=invis, width=0, height=0];
    llmt_pad2 [label="", style=invis, width=0, height=0];

    llmt_node1 -> llmt_node2 [label="yes"];
    llmt_node1 -> llmt_node3 [label="no"];
    llmt_node2 -> llmt_pad [style=invis];
    llmt_node3 -> llmt_pad2 [style=invis];
  }
}
'''

src = Source(dot_code, filename="four_trees", format="pdf")
src.render(cleanup=True)
print("✅ 成功生成 four_trees.pdf，可在当前目录查看。")
