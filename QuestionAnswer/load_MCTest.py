import xml.etree.ElementTree as ET

def load_MCTest(filename):

    tree = ET.parse(filename)
    root = tree.getroot()
    premise_2_hypolist={}
    for pair in root.findall('pair'):
        premise = pair.find('t').text
        hypolist = premise_2_hypolist.get(premise)
        if hypolist is None:
            hypolist = []
        hypothesis = pair.find('h').text
        label = pair.attrib.get('entailment')
        hypolist.append((hypothesis, label))
        premise_2_hypolist[premise] = hypolist

    print('Find ', len(premise_2_hypolist), ' documents')
    return premise_2_hypolist
if __name__ == "__main__":
    load_MCTest()
