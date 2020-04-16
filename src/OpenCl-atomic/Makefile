.DEFAULT_GOAL := all

KERNELS =  points2image euclidean_cluster ndt_mapping

.PHONY: clean $(KERNELS)

all: $(KERNELS)
	

$(KERNELS):
	$(MAKE) -C $@

clean:
	@for datei in $(KERNELS); do \
          $(MAKE) -C $$datei clean; \
        done
