###############################################################################
# Definitions
###############################################################################
FOLDER_ROOT=..
FOLDER_BUILD=../build

###############################################################################
# Modules
###############################################################################
MODULES=mm_allocator \
        profiler_counter \
        profiler_timer

SRCS=$(addsuffix .c, $(MODULES))
OBJS=$(addprefix $(FOLDER_BUILD)/, $(SRCS:.c=.o))

###############################################################################
# Rules
###############################################################################
all: $(OBJS)

# General building rule
$(FOLDER_BUILD)/%.o : %.c
	$(CC) $(CC_FLAGS) -I$(FOLDER_ROOT) -c $< -o $@