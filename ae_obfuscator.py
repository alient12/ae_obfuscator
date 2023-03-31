import re
import random
import string

VAR_FIRST_CHARS = "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
VAR_CHARS = "0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
B4_VAR_CHARS = " \t\n+-*/%!^&|=[({,:<>"
AFTER_VAR_CHARS = " \n+-*/%!^&|=[](),.:<>" + r"{}"
NOT_EXCEPTABLE_CHARS_B4_EQL = r"[=+-*/^%({})%|]" + "<>"


class block:
    def __init__(
        self,
        start=None,
        end=None,
        level=-1,
        block_type="",
        name="",
        first_line="",
        block_contents=[],
    ):
        self.start = start
        self.end = end
        self.level = level
        self.block_type = block_type
        self.name = name
        self.first_line = first_line
        self.block_contents = block_contents
        self.input_vars = []
        self.vars = []

    def setup_from_list(self, input_list):
        self.start = input_list[0]
        self.end = input_list[1]
        self.level = input_list[2]
        self.block_type = input_list[3]
        self.name = input_list[4]
        self.first_line = input_list[5]

    def show(self):
        print(self.level * 4 * " ", self.block_type, self.name, self.first_line)
        for l in self.block_contents:
            l.show()

    def show_with_vars(self):
        print(self.level * 4 * " ", self.block_type, self.name, self.first_line)
        if len(self.vars):
            print((self.level + 1) * 4 * " ", self.input_vars)
            print((self.level + 1) * 4 * " ", self.vars)
        for l in self.block_contents:
            l.show_with_vars()


def find_tabs_map(text):
    lines_tabs_num = []

    for line in text.split("\n"):
        tab_cnt = 0
        try:
            char = line[4 * tab_cnt : 4 * (tab_cnt + 1)]
        except Exception:
            continue
        while True:
            if char == "    ":
                tab_cnt += 1
            else:
                break
            try:
                char = line[4 * tab_cnt : 4 * (tab_cnt + 1)]
            except Exception:
                break
        lines_tabs_num.append(tab_cnt)
    return lines_tabs_num


def find_raw_structure(text, tabs_map):
    stack = []
    for i, line in line_generator(text):
        if len(line) == 0:
            continue
        scope_level = tabs_map[i]
        spaces_num = 4 * scope_level
        try:
            last_scope_level = stack[-1][2]
        except Exception:
            last_scope_level = -1
        if line.startswith(spaces_num * " " + "def "):
            if last_scope_level >= scope_level:
                for _ in range(last_scope_level - scope_level + 1):
                    start = stack.pop()
                    start[1] = i - 1
                    yield (start)
            func_name = line[spaces_num + len("def ") : line.index("(")]
            content = line[line.index("(") :]
            stack.append([i, -1, scope_level, "func", func_name, content])

        elif line.startswith(spaces_num * " " + "class "):
            if last_scope_level >= scope_level:
                for _ in range(last_scope_level - scope_level + 1):
                    start = stack.pop()
                    start[1] = i - 1
                    yield (start)
            if "(" in line:
                class_name = line[spaces_num + len("class ") : line.index("(")]
                content = line[line.index("(") : line.index(":")]
            else:
                class_name = line[spaces_num + len("class ") : line.index(":")]
                content = ""
            stack.append([i, -1, scope_level, "class", class_name, content])

        elif line.startswith(spaces_num * " " + "if "):
            if last_scope_level >= scope_level:
                for _ in range(last_scope_level - scope_level + 1):
                    start = stack.pop()
                    start[1] = i - 1
                    yield (start)
            content = line[line.index("f") + 2 : -1]
            stack.append([i, -1, scope_level, "if", "", content])

        elif line.startswith(spaces_num * " " + "elif "):
            if last_scope_level >= scope_level:
                for _ in range(last_scope_level - scope_level + 1):
                    start = stack.pop()
                    start[1] = i - 1
                    yield (start)
            content = line[line.index("f") + 2 : -1]
            stack.append([i, -1, scope_level, "elif", "", content])

        elif line.startswith(spaces_num * " " + "else:"):
            if last_scope_level >= scope_level:
                for _ in range(last_scope_level - scope_level + 1):
                    start = stack.pop()
                    start[1] = i - 1
                    yield (start)
            content = ""
            stack.append([i, -1, scope_level, "else", "", content])

        elif line.startswith(spaces_num * " " + "for "):
            if last_scope_level >= scope_level:
                for _ in range(last_scope_level - scope_level + 1):
                    start = stack.pop()
                    start[1] = i - 1
                    yield (start)
            content = line[line.index("r") + 2 : -1]
            stack.append([i, -1, scope_level, "for", "", content])

        elif line.startswith(spaces_num * " " + "while "):
            if last_scope_level >= scope_level:
                for _ in range(last_scope_level - scope_level + 1):
                    start = stack.pop()
                    start[1] = i - 1
                    yield (start)
            content = line[line.index("e") + 2 : -1]
            stack.append([i, -1, scope_level, "while", "", content])

        else:
            if last_scope_level >= scope_level:
                for _ in range(last_scope_level - scope_level + 1):
                    start = stack.pop()
                    start[1] = i - 1
                    yield (start)
    while len(stack) > 0:
        start = stack.pop()
        start[1] = i
        yield (start)


def make_structure(raw_structure):
    def get_lowest_rank(structure, rank):
        no_rank_flag = True
        sub_structures = []
        last_slice = 0
        for i, elem in enumerate(structure):
            if elem[2] == rank:
                no_rank_flag = False
                temp = block()
                temp.setup_from_list(structure[i])
                returned_structure = get_lowest_rank(structure[last_slice:i], rank + 1)
                temp.block_contents = returned_structure
                sub_structures.append(temp)
                last_slice = i + 1
        if no_rank_flag:
            return structure
        else:
            return sub_structures

    structure = get_lowest_rank(raw_structure, 0)
    return structure


def find_variables_old(text, paran):
    var_names = []
    equal_indices = [i.start() for i in re.finditer("=", text)]
    for idx in equal_indices:
        if text[idx + 1] == "=":
            continue
        i = idx - 1
        vars = ""
        while not text[i] == "\n":
            vars = text[i] + vars
            i -= 1
        if "=" in vars:
            continue
        if any(elem in vars for elem in NOT_EXCEPTABLE_CHARS_B4_EQL):
            continue
        if "." in vars:
            if not "self." in vars:
                continue
        for var in vars.split(","):
            var = var.replace(" ", "")
            if not var in var_names:
                add_var_flag = True
                for min, max in paran:
                    if min < i < max:
                        add_var_flag = False
                        break
                if add_var_flag:
                    var_names.append(var)

    for_vars = re.findall(r"for (.*?) in", text)
    for vars in for_vars:
        for var in vars.split(","):
            var = var.replace(" ", "")
            if not var in var_names:
                var_names.append(var)
    return var_names


def reform_text(text):
    reformed_lines_list = []
    line_is_open = False
    paranthes_flag = False
    start = -1
    reformed_line = ""
    for i, line in enumerate(text.split("\n")):
        if len(line) == 0:
            continue
        temp_line = line.replace(" ", "")
        if line_is_open:
            if (
                temp_line.endswith(")")
                or temp_line.endswith("]")
                or temp_line.endswith("}")
                or (paranthes_flag and temp_line.endswith(":"))
            ):
                line_is_open = False
                paranthes_flag = False
                cnt = 0
                while line[cnt] == " ":
                    cnt += 1
                if reformed_line.endswith(","):
                    reformed_line = reformed_line[:-1]
                reformed_line += line[cnt:]
                reformed_lines_list.append((start, i, reformed_line))
                reformed_line = ""
                continue
            else:
                cnt = 0
                while line[cnt] == " ":
                    cnt += 1
                reformed_line += line[cnt:]
        if not line_is_open:
            if (
                temp_line.endswith(",")
                or temp_line.endswith("[")
                or temp_line.endswith("{")
            ):
                start = i
                reformed_line += line
                line_is_open = True
            if temp_line.endswith("("):
                paranthes_flag = True
                start = i
                reformed_line += line
                line_is_open = True
    return reformed_lines_list


def line_generator(text):
    multilines_list = reform_text(text)
    start_list = []
    cnt = -1
    inside_multiline_flag = False
    for start, _end, _multiline in multilines_list:
        start_list.append(start)
    for i, line in enumerate(text.split("\n")):
        if len(line) == 0:
            continue
        if inside_multiline_flag:
            if i <= multilines_list[cnt][1]:
                continue
            else:
                inside_multiline_flag = False
        if i in start_list:
            cnt = start_list.index(i)
            inside_multiline_flag = True
            yield ((i, multilines_list[cnt][2]))
            continue
        yield ((i, line))


def line_generator_structure(text, structure):
    for i, line in enumerate(text.split("\n")):
        if len(line) == 0:
            continue
        inside_block_flag = False
        tab_slicer = 0
        for block in structure:
            if block.start <= i <= block.end:
                tab_slicer = 4 * (block.level + 1)
        ll = line[tab_slicer:]
        if ll.startswith("#"):
            continue
        yield ((i, ll))


def find_vars_line(input_line, var_names, instances_name="self"):
    paran = parenthetic_contents(input_line)
    try:
        idx = input_line.index("=")
        if input_line[idx + 1] == "=":
            return []
        vars = input_line[:idx]
        cnt = 0
        while vars[cnt] == " ":
            cnt += 1
        vars = vars[cnt:]
        if vars.startswith("#"):
            return []
        if any(elem in vars for elem in NOT_EXCEPTABLE_CHARS_B4_EQL):
            return []
        if "." in vars:
            not_a_var = True
            for instance_name in instances_name:
                if not vars.startswith(instance_name + "."):
                    not_a_var = False
                    break
            if not_a_var:
                return []
        if "," in vars:
            for var in vars.split(","):
                var = var.replace(" ", "")
                if not var in var_names:
                    add_var_flag = True
                    for min, max in paran:
                        if min < idx < max:
                            add_var_flag = False
                            break
                    if add_var_flag:
                        var_names.append(var)
        else:
            var = vars.replace(" ", "")
            if not var in var_names:
                add_var_flag = True
                for min, max in paran:
                    if min < idx < max:
                        add_var_flag = False
                        break
                if add_var_flag:
                    var_names.append(var)
    except Exception:
        pass

    for_vars = re.findall(r"for (.*?) in", input_line)
    for vars in for_vars:
        for var in vars.split(","):
            var = var.replace(" ", "")
            if not var in var_names:
                var_names.append(var)
    return var_names


def find_variables(text, structure):
    global_vars = []
    instances_names = find_class_instances(raw_text, structure)
    instances_names.append("self")

    # find global variables
    for i, line in enumerate(text.split("\n")):
        if len(line) == 0:
            continue
        inside_block_flag = False
        tab_slicer = 0
        for block in structure:
            if block.start <= i <= block.end:
                if block.block_type in ["func", "class", "for", "while"]:
                    inside_block_flag = True
                    break
                elif block.start == i:
                    inside_block_flag = True
                    break
                tab_slicer = 4 * (block.level + 1)
        if inside_block_flag:
            continue

        ll = line[tab_slicer:]
        if ll.startswith("#"):
            continue
        vv = find_vars_line(ll, global_vars)
        global_vars = vv

    def find_func_input_vars(input_line, input_vars_name):
        try:
            vars = input_line[input_line.index("(") + 1 : input_line.rindex(")")]
        except Exception:
            return
        vars = vars.split(",")
        for var in vars:
            var = var.replace(" ", "")
            try:
                var = var[: var.index("=")]
            except Exception:
                pass
            if var == "self":
                continue
            input_vars_name.append(var)

    # find local variables
    def find_local_vars(block):
        if not block.block_type in ["if", "elif", "else"]:
            for i, line in line_generator(text):
                if block.start > i or block.end < i:
                    continue
                if len(line) == 0:
                    continue
                tab_slicer = 4 * (block.level + 1)

                if block.start == i and block.block_type == "func":
                    find_func_input_vars(line, block.input_vars)
                    tab_slicer = 4 * (block.level)

                inside_block_flag = False

                for inner_block in block.block_contents:
                    #### fuck
                    if isinstance(inner_block, list):
                        print(type(block))
                        block.show()
                    #### fuck
                    if inner_block.start <= i <= inner_block.end:
                        if inner_block.block_type in ["func", "class", "for", "while"]:
                            inside_block_flag = True
                            break
                        elif inner_block.start == i:
                            inside_block_flag = True
                            break

                        # inner blocks of (if, elif, else)
                        for starter in ["def ", "class ", "for ", "while "]:
                            if starter in line:
                                inside_block_flag = True
                                break
                        if inside_block_flag:
                            break
                        tab_slicer = 4 * (block.level + inner_block.level + 1)
                if inside_block_flag:
                    continue

                ll = line[tab_slicer:]
                if ll.startswith("#"):
                    continue

                find_vars_line(ll, block.vars)

        for i in range(len(block.block_contents)):
            find_local_vars(block.block_contents[i])

    for i in range(len(structure)):
        if structure[i].block_type in ["if", "elif", "else"]:
            continue
        find_local_vars(structure[i])

    return global_vars, structure


def find_class_instances(text, structure):
    classes_list = []
    instances_name = []
    for block in structure:
        if block.block_type == "class":
            classes_list.append((block.name, block.end))

    for class_name, start_line in classes_list:
        for i, line in line_generator_structure(text, structure):
            if start_line < i:
                if class_name in line:
                    instances_name = find_vars_line(line, instances_name)

    return instances_name


def unpack(d):
    for k, v in d.items():
        tmp = []
        for vv in v:
            tmp.append(vv)
        yield (k, tmp)


def replace_vars_old(text, var_names, paran):
    vars_inices = {}
    for var in var_names:
        var_inices = []
        indices = [i.start() for i in re.finditer(var, text)]
        for idx in indices:
            if text[idx - 1] in B4_VAR_CHARS:
                if idx + len(var) == len(text):
                    var_inices.append(idx)
                elif text[idx + len(var)] in AFTER_VAR_CHARS:
                    for min, max in paran:
                        add_var_flag = True
                        if min < idx < max:
                            _ = idx + len(var)
                            temp = text[_ : _ + 4]
                            if "=" in temp and not any(elem in temp for elem in "<!>"):
                                add_var_flag = False
                                break
                    if add_var_flag:
                        var_inices.append(idx)
        vars_inices[var] = var_inices
    var_cnt = 0
    text_list = list(text)
    for var, indices in unpack(vars_inices):
        first_letter = "".join(
            random.SystemRandom().choice(
                string.ascii_uppercase + string.ascii_lowercase
            )
            for _ in range(69)
        )
        new_var = (
            first_letter
            + str(var_cnt)
            + "".join(
                random.SystemRandom().choice(
                    string.ascii_uppercase + string.ascii_lowercase + string.digits
                )
                for _ in range(69)
            )
        )
        if var[:5] == "self.":
            new_var = "self." + new_var
        for idx in indices:
            for _ in range(1, len(var)):
                text_list[idx + _] = ""
            # new_var = "Asghar"
            text_list[idx] = new_var + str(var_cnt)
        var_cnt += 1
    return "".join(text_list)


def parenthetic_contents(text):
    """Generate parenthesized contents in text as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(text):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            start = stack.pop()
            # yield (len(stack), text[start + 1 : i])
            yield ((start, i))


# def check_inside_paranthesis(text):


# def rename_moduls(text):
#     indices = [i.start() for i in re.finditer("import", text)]


if __name__ == "__main__":
    filename_without_extension = "contrast"
    # filename_without_extension = "ae_detect"
    # filename_without_extension = "ae_smoke_detector"
    # filename_without_extension = "ae_obfuscator_copy"
    with open(filename_without_extension + ".py") as file:
        raw_text = file.read()
    # paran = list(parenthetic_contents(raw_text))
    # var_names = find_variables(raw_text, paran)
    # text = replace_vars(raw_text, var_names, paran)
    tabs_map = find_tabs_map(raw_text)
    # print(tabs_map)
    raw_structure = list(find_raw_structure(raw_text, tabs_map))
    structure = make_structure(raw_structure)
    # for st in structure:
    #     st.show()
    global_vars, structure = find_variables(raw_text, structure)
    print(global_vars)
    for st in structure:
        st.show_with_vars()

    # instances = find_class_instances(raw_text, structure)
    # print(instances)
    # print(reform_text(raw_text))

    # with open(filename_without_extension + "_obf" + ".py", "w") as file:
    #     file.write(text)
